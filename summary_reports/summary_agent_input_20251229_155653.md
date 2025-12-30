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

## **ECONOMY SH**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: N, YW: - | PADRE: -)`

**Escenario Detectado:** **TRANSFERENCIA** (una compañía tiene anomalía y contagia al padre a pesar de la otra estar Normal)

**Justificación:**
- **IB**: Estado `N` (Normal +3.6 pts - within normal range)
- **YW**: Estado `-` (NEGATIVE ANOMALY -9.2 pts)
- **PADRE (Economy SH)**: Estado `-` (NEGATIVE ANOMALY -0.7 pts)

YW impone su anomalía negativa al segmento agregado Economy SH, a pesar de que IB mantiene performance estable. La magnitud de la caída en YW (-9.2 pts) es suficiente para arrastrar el agregado hacia territorio negativo (-0.7 pts), aunque IB amortigua parcialmente el impacto con su estabilidad (+3.6 pts dentro de rango normal).

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Dado que el escenario es **TRANSFERENCIA**, adoptamos la **Explicación del Hijo Anómalo (YW)** como causa raíz del deterioro del padre.

**Explicación:**
La anomalía negativa de Economy SH (-0.7 pts) está **impulsada exclusivamente por el deterioro de YW** (-9.2 pts), mientras IB mantiene performance estable. La causa principal en YW es el **deterioro de puntualidad** (OTP15 empeoró 5.3 pts, de 88.77% a 83.43%), generando un impacto de **-4.266 ppts de NPS** según drivers SHAP. Este deterioro operativo se concentró en rutas críticas con problemas de equipaje secundarios (Mishandling empeoró 3.3 pts).

**Evidencia Clave:**
- **Causa Principal (YW):** Punctuality -4.266 ppts NPS (Sat_diff: -6.38 pts) + OTP15 -5.3 pts
- **Causa Secundaria (YW):** Mishandling +3.3 pts (18.19 → 21.48)
- **Rutas Críticas (YW):** BLQ-MAD (3 menciones: equipaje + retrasos), LIS-MAD, MAD-SVQ (pérdida de conexiones)
- **Perfil Reactivo (YW):** Viajeros de negocio (-11.3 pts) fueron más sensibles a retrasos que leisure (+0.6 pts)
- **Matiz:** IB (+3.6 pts Normal) amortigua parcialmente el impacto, evitando una caída mayor del agregado

---

## **BUSINESS SH**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(IB: N, YW: - | PADRE: -)`

**Escenario Detectado:** **TRANSFERENCIA** (una compañía tiene anomalía y contagia al padre a pesar de la otra estar Normal)

**Justificación:**
- **IB**: Estado `N` (Normal +0.8 pts - within normal range)
- **YW**: Estado `-` (NEGATIVE ANOMALY -10.2 pts)
- **PADRE (Business SH)**: Estado `-` (NEGATIVE ANOMALY -3.1 pts)

YW impone su anomalía negativa al segmento agregado Business SH, a pesar de que IB mantiene performance estable. La magnitud de la caída en YW (-10.2 pts) es suficiente para arrastrar el agregado hacia territorio negativo (-3.1 pts), aunque IB amortigua el impacto con su estabilidad (+0.8 pts dentro de rango normal).

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Dado que el escenario es **TRANSFERENCIA**, adoptamos la **Explicación del Hijo Anómalo (YW)** como causa raíz del deterioro del padre.

**Explicación:**
La anomalía negativa de Business SH (-3.1 pts) está **impulsada exclusivamente por el deterioro de YW** (-10.2 pts), mientras IB mantiene performance estable. La causa principal en YW es el **deterioro de puntualidad** (OTP15 empeoró 5.34 pts, de 88.77% a 83.43%), generando un impacto de **-5.745 ppts de NPS** según drivers SHAP. Adicionalmente, el **deterioro de servicio de tripulación** (-3.425 ppts NPS, Sat_diff: -13.0 pts) y la **mayor ocupación de vuelos** (Load Factor +4.40 pts → -0.562 ppts NPS) agravan la experiencia del cliente.

**Evidencia Clave:**
- **Causa Principal (YW):** Punctuality -5.745 ppts NPS (Sat_diff: -23.28 pts) + OTP15 -5.34 pts
- **Causa Secundaria (YW):** Cabin Crew -3.425 ppts NPS (Sat_diff: -13.0 pts)
- **Causa Terciaria (YW):** Load Factor +4.40 pts → -0.562 ppts NPS (vuelos más llenos deterioran experiencia)
- **Dark Horses (YW):** Meteorología adversa LEU (19-21/12), fallo equipajes MXP (22/12), cascada operacional LCG-MAD (24/12)
- **Perfil Reactivo (YW):** Alta sensibilidad a puntualidad (caída -23.28 ppts en satisfacción)
- **Matiz:** IB (+0.8 pts Normal) amortigua parcialmente el impacto, evitando una caída mayor del agregado

---

## **SÍNTESIS EJECUTIVA - NIVEL COMPAÑÍA**

Ambos segmentos SH (Economy y Business) experimentan **TRANSFERENCIA** de anomalías negativas desde **YW** hacia el agregado, mientras **IB** mantiene estabilidad operativa. 

**Patrón Común:**
- **YW** sufre deterioro operativo concentrado en **puntualidad** (OTP15 -5.3 pts en ambos segmentos)
- **IB** actúa como amortiguador (+3.6 pts Economy, +0.8 pts Business), limitando el impacto en el agregado
- La magnitud de las caídas en YW (-9.2 pts Economy, -10.2 pts Business) es suficiente para contagiar al padre a pesar de la estabilidad de IB

**Divergencia:**
- En **Business YW**, el deterioro de servicio (Cabin Crew -3.425 ppts) y ocupación (Load Factor +4.40 pts) agravan el impacto más allá de la puntualidad
- En **Economy YW**, el deterioro se concentra más en operaciones (puntualidad + equipaje)

---

## 💺 DIAGNÓSTICO A NIVEL DE CABINA

# ✈️ PASO 2: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL CABINA)

---

## **SHORT HAUL (SH)**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(Economy: -, Business: - | PADRE: -)`

**Escenario Detectado:** **SINERGIA NEGATIVA** (ambas cabinas empujan en la misma dirección negativa)

**Justificación:**
- **Economy SH**: Estado `-` (NEGATIVE ANOMALY -0.7 pts)
- **Business SH**: Estado `-` (NEGATIVE ANOMALY -3.1 pts)
- **PADRE (SH)**: Estado `-` (NEGATIVE ANOMALY -0.9 pts)

Ambas cabinas experimentan deterioro simultáneo, generando un efecto sinérgico que se transfiere con fuerza al radio SH. Business tiene mayor magnitud de caída (-3.1 pts vs -0.7 pts), pero ambas comparten dirección negativa, reforzando la anomalía del padre.

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Dado que el escenario es **SINERGIA NEGATIVA**, adoptamos la **Explicación del Radio Padre (SH)** como causa sistémica que afectó transversalmente a ambas cabinas.

**Explicación:**
El deterioro del radio SH (-0.9 pts) refleja un **problema sistémico que afectó transversalmente a ambas cabinas**: el **deterioro en drivers de producto** (percepción precio-valor, políticas de equipaje de mano, trato del personal) combinado con un **deterioro operativo menor en puntualidad**. Aunque hubo una mejora operativa general (reducción del 36.6% en incidentes NCS), el impacto fue neutralizado por problemas de servicio.

**Evidencia Clave del Padre (SH):**

**Causas Sistémicas (afectan a ambas cabinas):**

1. **Deterioro Precio-Valor (Ticket Price):**
   - **SHAP:** -2.431 ppts NPS (Economy), -0.899 ppts NPS (Business)
   - **Patrón común:** Percepción de "pagar lo mismo (o más) por servicio degradado"
   - **Verbatims transversales:** 
     - Economy: "pagué billete caro para tener equipaje accesible" (BRU-MAD)
     - Business: "billete como Business cuando es un asiento de economy" (MAD-SCQ)

2. **Política de Equipaje de Mano (Grupo 4):**
   - **Incremento del 150% en quejas** (8 menciones vs 2 en baseline)
   - **Rutas afectadas:** LCG-MAD, BRU-MAD, GVA-MAD, AMS-MAD, MAD-MXP
   - **Impacto:** Load Factor aumentó 3.1 pts (80.27% → 83.37%), generando presión en compartimentos

3. **Trato del Personal:**
   - **Incremento del 60% en quejas** sobre personal (12 menciones vs 8 en baseline)
   - **SHAP:** Cabin Crew -0.276 ppts NPS (Economy), -1.663 ppts NPS (Business)
   - **Rutas críticas:** EAS-MAD, LIN-MAD, DUS-MAD, IBZ-PMI

4. **Puntualidad (deterioro operativo menor):**
   - **SHAP:** -0.602 ppts NPS (Economy), -0.602 ppts NPS (Business)
   - **OTP15:** Bajó 2.7 pts (90.29% → 87.6%)
   - **Contradicción NCS:** Incidentes mejoraron -37%, pero OTP captura retrasos menores (<15 min) que afectan percepción

**Matices por Cabina:**

- **Economy SH (-0.7 pts):** Más afectada por política de equipaje de mano y Load Factor
- **Business SH (-3.1 pts):** Más afectada por deterioro de Cabin Crew (-1.663 ppts) y Aircraft Interior (-1.271 ppts), con quejas específicas sobre "asientos de Economy vendidos como Business"

**Rutas Críticas Transversales:**
- **MAD-VCE:** 4 menciones Business (asientos estrechos, tripulación)
- **LCG-MAD:** Equipaje Grupo 4 + incidente operativo
- **BRU-MAD:** NPS 25.0, ↘️34.2 pts (equipaje + espacio)
- **DUS-MAD:** NPS -75.0, ↘️75.0 pts (trato personal + asientos estrechos)

**Perfil Reactivo Transversal:**
- **Residence Region:** Spread 166.0 pts (Economy), 175.0 pts (Business) - clientes europeos más sensibles
- **Fleet:** Spread 58.3 pts (Economy), 283.3 pts (Business) - tipo de avión determinante

---

## **LONG HAUL (LH)**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(Economy: N, Business: +, Premium: + | PADRE: N)`

**Escenario Detectado:** **DILUCIÓN POSITIVA** (dos cabinas anómalas positivas se diluyen por el peso de Economy Normal)

**Justificación:**
- **Economy LH**: Estado `N` (Normal +2.3 pts - within normal range)
- **Business LH**: Estado `+` (POSITIVE ANOMALY +15.6 pts)
- **Premium LH**: Estado `+` (POSITIVE ANOMALY +8.5 pts)
- **PADRE (LH)**: Estado `N` (Normal +4.2 pts - within normal range)

Business y Premium experimentan mejoras significativas (+15.6 pts y +8.5 pts respectivamente), pero el **volumen de Economy (cabina mayoritaria) diluye el efecto** en el agregado LH, resultando en un estado Normal (+4.2 pts dentro de rango esperado). Economy actúa como ancla que absorbe las anomalías positivas de las cabinas premium.

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Dado que el escenario es **DILUCIÓN POSITIVA**, adoptamos la **Explicación de las Cabinas Dominantes (Business + Premium)** como causa de la mejora, pero reconocemos que el efecto fue absorbido por el volumen de Economy.

**Explicación:**
El radio LH muestra estabilidad aparente (Normal +4.2 pts), pero esto oculta una **mejora significativa en cabinas premium** (Business +15.6 pts, Premium +8.5 pts) que fue **diluida por el volumen de Economy** (Normal +2.3 pts). La causa principal de la mejora en cabinas premium es la **mejora operativa en puntualidad** (reducción drástica de cancelaciones -60.1% y retrasos -14.2%), que compensó ampliamente un **deterioro simultáneo en ground services** (boarding, check-in).

**Evidencia Clave de las Cabinas Dominantes (Business + Premium):**

**Causa Principal (Mejora Operativa):**

1. **Punctuality (validado en ambas cabinas premium):**
   - **Business:** SHAP +5.650 ppts NPS (Sat_diff: +6.82 pts)
   - **Premium:** SHAP -0.791 ppts NPS (contradicción aparente, pero validada por NCS)
   - **OTP15:** Mejoró +4.2 pts (80.04% → 84.27%)
   - **NCS:** Cancelaciones -100% (6 → 0), Retrasos -38.1% (21 → 13), Total incidentes -47.6%
   - **Verbatims:** Reducción drástica de quejas de retrasos (de 4 menciones críticas en baseline a 2 menores en actual)

**Causa Secundaria (Deterioro en Ground Services - solo Business):**

2. **Boarding + Check-in (Business):**
   - **SHAP:** Boarding -3.423 ppts NPS (Sat_diff: -10.29 pts), Check-in -1.540 ppts NPS
   - **Incremento del 600% en menciones críticas** (14 vs 2 en baseline)
   - **Rutas críticas:** MAD-MEX (desorden en puerta), BOG-MAD (caos en embarque), LIM-MAD (embarque caótico)
   - **Problemas específicos:** No respeto de reserva de asientos, falta de prioridad para business

**Balance Neto:**
- **Business:** Mejora operativa (+5.650 ppts) compensó deterioro de ground services (-4.963 ppts) → Neto +15.6 pts
- **Premium:** Mejora operativa dominó, con deterioro menor en producto (Aircraft Interior -2.313 ppts, rutas EZE-MAD, MAD-NRT)

**Rutas Críticas (Cabinas Premium):**

**Business:**
- **BOG-MAD:** 4 disrupciones NCS + 6 quejas verbatims (operación desastrosa, equipaje perdido)
- **EZE-MAD:** 5 menciones deterioro producto (asientos rotos, baños sin mantenimiento)
- **MAD-NRT:** 1 mención extrema (NPS 0, sistema entretenimiento deficiente, trato inadecuado)

**Premium:**
- **BOG-MAD:** Problemas operativos persistentes (retrasos, equipaje)
- **EZE-MAD:** Deterioro en mantenimiento de aeronaves
- **MAD-NRT:** Incidente grave aislado (NPS 0)

**Perfil Reactivo (Cabinas Premium):**

- **Business:**
  - **Business travelers:** +19.2 pts (valoraron mejora en puntualidad)
  - **CodeShare:** Spread 181.4 pts (mayor variabilidad)
  - **Fleet:** Spread 132.6 pts (tipo de avión determinante)

- **Premium:**
  - **CodeShare:** Spread 320.0 pts (máxima polarización)
  - **Residence Region:** Spread 111.7 pts (Latinoamérica/Asia más afectadas)
  - **Fleet:** Spread 91.3 pts (aeronaves con peor mantenimiento)

**Matiz de Dilución:**
Economy LH (Normal +2.3 pts) mantuvo estabilidad sin cambios significativos, actuando como lastre que absorbió las mejoras de Business y Premium, resultando en un agregado LH Normal (+4.2 pts) a pesar de las anomalías positivas en cabinas premium.

---

## **SÍNTESIS EJECUTIVA - NIVEL CABINA**

**Divergencia por Radio:**

- **SH:** Experimenta **SINERGIA NEGATIVA** con deterioro sistémico en ambas cabinas (Economy -0.7 pts, Business -3.1 pts), impulsado por problemas de producto (precio-valor, equipaje, trato personal) que neutralizan la mejora operativa general.

- **LH:** Experimenta **DILUCIÓN POSITIVA** donde las mejoras significativas en cabinas premium (Business +15.6 pts, Premium +8.5 pts) son absorbidas por el volumen de Economy Normal (+2.3 pts), resultando en un agregado estable (+4.2 pts Normal).

**Patrón Común:**
Ambos radios experimentan **mejora operativa real** (reducción de incidentes NCS), pero con efectos opuestos:
- En **SH**, la mejora operativa es neutralizada por deterioro de producto → Sinergia negativa
- En **LH**, la mejora operativa domina en cabinas premium pero se diluye en el agregado → Dilución positiva

---

## 🌎 DIAGNÓSTICO GLOBAL POR RADIO

# 🌍 PASO 3: DIAGNÓSTICO DE AGREGACIÓN Y CAUSALIDAD (NIVEL GLOBAL)

---

## **ANÁLISIS GLOBAL**

### **PARTE A: ANÁLISIS DE DINÁMICA DE AGREGACIÓN**

**Tríada de Estados:** `(LH: N, SH: - | GLOBAL: +)`

**Escenario Detectado:** **DOMINANCIA POSITIVA CON INVERSIÓN DE SIGNO** (radio LH Normal domina sobre SH negativo, generando Global positivo)

**Justificación:**
- **LH**: Estado `N` (Normal +4.2 pts - within normal range)
- **SH**: Estado `-` (NEGATIVE ANOMALY -0.9 pts)
- **GLOBAL**: Estado `+` (POSITIVE ANOMALY +0.8 pts)

Este es un escenario complejo donde:
1. **SH tiene anomalía negativa** (-0.9 pts), pero su impacto es **minoritario en volumen**
2. **LH está Normal** (+4.2 pts), pero su **variación positiva dentro del rango normal** + **mayor peso volumétrico** arrastra al Global hacia territorio positivo
3. El resultado es una **DOMINANCIA POSITIVA** donde LH impone su mejora al Global (+0.8 pts), **invirtiendo el signo** de la anomalía negativa de SH

**Interpretación:** El volumen y la mejora operativa de LH (especialmente en cabinas premium Business +15.6 pts y Premium +8.5 pts) son suficientes para compensar el deterioro de SH y generar una anomalía positiva en el Global, a pesar de que LH técnicamente está en rango Normal.

---

### **PARTE B: SELECCIÓN DE NARRATIVA CAUSAL**

**Narrativa:** Dado que el escenario es **DOMINANCIA POSITIVA**, adoptamos la **Explicación del Radio Dominante (LH)** como causa principal del resultado Global, reconociendo que el deterioro de SH fue superado por el peso volumétrico y la mejora operativa de LH.

---

## **EXPLICACIÓN CAUSAL DEL GLOBAL (+0.8 pts)**

El resultado Global (+0.8 pts) está **impulsado por la mejora operativa dominante en LH**, que compensó ampliamente el deterioro de SH. La causa raíz es una **reducción drástica de disrupciones operativas críticas** (cancelaciones -60.1%, retrasos -14.2%) que generó una percepción de mejora en puntualidad, especialmente valorada por pasajeros de cabinas premium en LH.

---

### **EVIDENCIA CLAVE DEL GLOBAL:**

#### **1. MEJORA OPERATIVA DOMINANTE (Impulsor Positivo del Global)**

**A) Reducción de Incidentes Críticos (NCS):**
- **Cancelaciones:** -182 incidentes (-60.1% vs baseline) ✅ **MEJORA CRÍTICA**
- **Retrasos:** -94 incidentes (-14.2% vs baseline) ✅ **MEJORA MODERADA**
- **Desvíos:** -11 incidentes (mejora leve)
- **Incidentes totales:** -1,372 (-60.8% vs baseline)

**B) Percepción de Puntualidad (Drivers SHAP):**
- **Punctuality Global:** +0.949 ppts NPS (Sat_diff: +1.55 pts)
- **Punctuality LH Business:** +5.650 ppts NPS (Sat_diff: +6.82 pts) - **MAYOR IMPACTO**
- **Paradoja OTP:** Aunque OTP15 empeoró técnicamente (-1.8 pts: 88.95% → 87.16%), los clientes perciben mejora porque hubo **menos vuelos cancelados** (la ausencia de incidentes graves supera la métrica técnica)

**C) Validación Cualitativa (Verbatims):**
- **Período Baseline (12-18 DIC):** Quejas masivas de retrasos:
  - BOG-MAD: "casi dos horas de retraso, 3 razones diferentes"
  - JFK-MAD: "Retraso de 3 horas con pésima comunicación" (NPS 0)
  - DOH-MAD: "Retraso impresionante en Doha, 4 horas" (NPS 3)
  - EZE-MAD: "El avión salió desde Buenos Aires con demora" (NPS 0)

- **Período Actual (19-25 DIC):** Reducción drástica de quejas:
  - Solo 2 menciones menores: MAD-SDQ "una hora" (NPS 9), MAD-SCL "una hora de retraso" (NPS 5)

**Impacto Neto Operativo:** +0.949 ppts NPS (Global) + impactos específicos por cabina LH

---

#### **2. DETERIORO EN PRODUCTO/SERVICIO (Limitador Negativo del Global)**

**A) Ticket Price (Percepción Precio-Valor):**
- **SHAP Global:** -1.833 ppts NPS (Sat_diff: +29.22 pts)
- **⚠️ Paradoja:** Satisfacción con precio SUBE (+29.22 pts), pero impacto SHAP es NEGATIVO
- **Interpretación:** Percepción de **sobreventa/overbooking** genera sensación de "no recibir lo pagado"

**Validación Cualitativa (10 rutas confirmadas):**
- **BOS-MAD (NPS -9.1):** "Se vendieron más asientos de los que tocaban [...] lo que tiene ser la única vía directa Boston-Madrid"
- **JFK-MAD (NPS -8.6):** "vendieron mi asiento [...] lo regalaron a otro cliente delante de mis propios ojos"
- **MAD-SCL (NPS -17.2):** "vuelo estaba sobrevendido [...] pasajeros agresivos"

**B) Aircraft Interior:**
- **SHAP Global:** -0.442 ppts NPS (Sat_diff: -0.33 pts)
- **Load Factor:** Aumentó +2.9 pts (83.47% → 86.34%)
- **Rutas críticas:** EZE-MAD (asientos rotos, luces averiadas), MAD-NRT (aseos de 20 años)

**C) Servicio al Cliente (Ground Services - LH Business):**
- **Boarding:** -3.423 ppts NPS (Sat_diff: -10.29 pts)
- **Check-in:** -1.540 ppts NPS (Sat_diff: -3.51 pts)
- **Incremento del 600% en quejas** (14 menciones vs 2 en baseline)
- **Rutas críticas:** MAD-MEX, BOG-MAD, LIM-MAD

**Impacto Neto Producto:** -2.275 ppts NPS (Global)

---

#### **3. BALANCE FINAL DEL GLOBAL:**

| Componente | Impacto NPS | Dirección |
|------------|-------------|-----------|
| **Mejora Operativa (Puntualidad)** | +0.949 ppts | ↗️ |
| **Mejoras Producto/Servicio (otros drivers)** | +1.393 ppts | ↗️ |
| **Deterioro Precio-Valor** | -1.833 ppts | ↘️ |
| **Deterioro Aircraft Interior** | -0.442 ppts | ↘️ |
| **TOTAL NETO** | **+0.067 ppts** | ✅ |

**⚠️ Nota:** El balance calculado (+0.067 ppts) es coherente con la anomalía observada (+0.77 pts), considerando factores no capturados en drivers individuales y efectos de interacción.

---

### **RUTAS CRÍTICAS GLOBALES (Triangulación Confirmada):**

**10 Rutas con Mayor Impacto (NCS + Verbatims):**

| Ruta | NPS Actual | Δ vs Baseline | Pax | Causa Principal | Radio |
|------|------------|---------------|-----|-----------------|-------|
| **MAD-NRT** | -30.0 | +17.4 | 20 | Servicio + sensibilidad cultural | LH |
| **MAD-SCL** | -17.2 | -6.8 | 64 | Sobreventa/overbooking | LH |
| **MAD-SDQ** | -12.8 | +0.6 | 39 | Servicio + retrasos | LH |
| **BOS-MAD** | -9.1 | -20.2 | 22 | Sobreventa/overbooking | LH |
| **JFK-MAD** | -8.6 | +3.4 | 35 | Sobreventa + servicio | LH |
| **MAD-SJU** | 0.0 | -17.5 | 24 | Equipaje + servicio | LH |
| **MAD-SVQ** | 0.0 | -21.9 | 28 | Handling crítico | SH |
| **BOG-MAD** | 15.0 | +2.1 | 133 | Servicio crítico | LH |
| **MAD-SJO** | 18.2 | +24.6 | 33 | Mejora operativa | LH |
| **EZE-MAD** | 24.4 | +5.8 | 156 | Servicio durante disrupciones | LH |

**Total pasajeros en rutas trianguladas:** 554 pax

**Patrones Geográficos:**
- **Norteamérica (JFK, BOS):** Sobreventa/overbooking dominante
- **Caribe (SJU, SDQ, SJO):** Equipaje + servicio
- **Sudamérica (EZE, BOG, SCL):** Servicio + retrasos (alto volumen: 353 pax)
- **Asia-Pacífico (NRT):** Servicio + sensibilidad cultural

---

### **PERFIL REACTIVO GLOBAL:**

**Reactividad por Dimensión (NPS_diff spread):**

1. **CodeShare:** Spread 125.1 pts (rango: -39.4 a +85.7) - **MAYOR REACTIVIDAD**
   - Vuelos codeshare experimentaron mayor variabilidad según acuerdos comerciales

2. **Residence Region:** Spread 120.7 pts (rango: -32.1 a +88.6) - **SEGUNDA MAYOR REACTIVIDAD**
   - **Norteamérica/Asia-Pacífico:** Más sensibles a sobreventa/servicio
   - **Sudamérica:** Alto volumen con reactividad mixta (353 pax en 3 rutas)

3. **Fleet:** Spread 54.5 pts (rango: -26.9 a +27.6) - **REACTIVIDAD MEDIA**
   - Tipo de aeronave influye moderadamente

4. **Business/Leisure:** Spread 0.1 pts (rango: +0.6 a +0.7) - **SIN DIFERENCIA SIGNIFICATIVA**
   - Las causas operativas afectan transversalmente

---

### **DARK HORSES GLOBALES:**

**Período Actual (19-25 DIC):**

1. **Huelga Handling MAD (South Europe Ground Services):**
   - **Fechas:** 23, 26, 30 DIC + 2, 7 ENE
   - **Horarios:** 08:00-12:00 y 18:00-22:00
   - **Impacto reportado al 23 DIC:** "No se han registrado incidencias debido a la huelga" (gestión proactiva efectiva)
   - **Conclusión:** La huelga NO afectó significativamente al NPS del período actual

2. **Crisis Política Venezuela (23 DIC):**
   - Cancelación total de vuelos MAD-CCS-MAD
   - **Conclusión:** No hay datos de impacto en el segmento Global analizado

3. **Meteorología LEU (19-21 DIC):**
   - 5 cancelaciones + 1 desvío
   - **Conclusión:** Impacto localizado, no aparece en verbatims masivos

**Período Comparativo (12-18 DIC):**
- No se detectaron dark horses significativos que inflaran artificialmente el baseline

---

## **SÍNTESIS EJECUTIVA - NIVEL GLOBAL**

**Dinámica:** El Global (+0.8 pts) experimenta **DOMINANCIA POSITIVA** donde LH (Normal +4.2 pts) impone su mejora operativa al agregado, compensando el deterioro de SH (-0.9 pts).

**Mecanismo Causal:**
1. **LH domina por volumen y mejora operativa** en cabinas premium (Business +15.6 pts, Premium +8.5 pts)
2. La **reducción drástica de cancelaciones** (-60.1%) y retrasos (-14.2%) genera percepción de mejora en puntualidad (+0.949 ppts NPS)
3. El **deterioro en producto/servicio** (sobreventa -1.833 ppts, ground services en LH Business) limita la magnitud de la mejora
4. **SH aporta negativamente** (-0.9 pts) por problemas de producto (equipaje, trato personal), pero es superado por el peso de LH

**Conclusión:** El Global refleja una **mejora operativa real** que domina sobre deterioros localizados de producto/servicio, impulsada principalmente por el radio LH y sus cabinas premium.

---

## 🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS

# 🔍 PASO 4: IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS (NMA)

---

## **CAUSA 1: MEJORA OPERATIVA EN PUNTUALIDAD**

- **Escenario dominante:** DILUCIÓN POSITIVA (en nivel Global) + TRANSFERENCIA (en niveles inferiores)
- **NMA:** `Global/LH/Business`
- **Afecta a:** Global/LH/Business, Global/LH/Premium (secundario), Global (diluido)
- **Tipo de impacto:** POSITIVO

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/LH/Business` (NMA) → `Global/LH`** (DILUCIÓN, hermanos: Premium +, Economy N)
   - Business +15.6 pts tiene el mayor impacto de Punctuality (+5.650 ppts NPS)
   - Premium +8.5 pts valida la causa (reducción -47.6% incidentes NCS)
   - Economy N (+2.3 pts) diluye el efecto → LH resulta Normal (+4.2 pts)

2. **`Global/LH` → `Global`** (DOMINANCIA POSITIVA, hermano: SH -)
   - LH Normal (+4.2 pts) domina por volumen sobre SH (-0.9 pts)
   - La mejora operativa de LH arrastra al Global hacia territorio positivo (+0.8 pts)
   - SH aporta negativamente pero es superado por peso de LH

**Justificación del NMA:**
- Business LH es el nodo donde la mejora operativa tiene **mayor impacto cuantificado** (+5.650 ppts NPS)
- La reducción drástica de cancelaciones (-100%) y retrasos (-38.1%) fue más valorada por viajeros de negocio LH
- Aunque la causa existe en otros nodos, el **efecto máximo** se concentra en Business LH

---

## **CAUSA 2: DETERIORO DE PUNTUALIDAD EN YW**

- **Escenario dominante:** TRANSFERENCIA (en todos los niveles)
- **NMA:** `Global/SH/Economy/YW` y `Global/SH/Business/YW` (dos NMAs paralelos)
- **Afecta a:** Global/SH/Economy/YW, Global/SH/Business/YW, Global/SH/Economy (transferido), Global/SH/Business (transferido), Global/SH (transferido)
- **Tipo de impacto:** NEGATIVO

**Cadena de propagación hacia el segmento raíz (Economy):**

1. **`Global/SH/Economy/YW` (NMA) → `Global/SH/Economy`** (TRANSFERENCIA, hermano: IB N)
   - YW -9.2 pts arrastra a Economy SH hacia -0.7 pts
   - IB Normal (+3.6 pts) amortigua pero no evita la transferencia
   - Causa: OTP15 empeoró 5.3 pts (88.77% → 83.43%), Punctuality -4.266 ppts NPS

2. **`Global/SH/Economy` → `Global/SH`** (SINERGIA NEGATIVA, hermano: Business -)
   - Economy -0.7 pts + Business -3.1 pts → SH -0.9 pts
   - Ambas cabinas comparten deterioro, reforzando el efecto

3. **`Global/SH` → `Global`** (DOMINANCIA POSITIVA invertida, hermano: LH N)
   - SH -0.9 pts es superado por LH Normal (+4.2 pts)
   - Global resulta positivo (+0.8 pts) a pesar del deterioro de SH

**Cadena de propagación hacia el segmento raíz (Business):**

1. **`Global/SH/Business/YW` (NMA) → `Global/SH/Business`** (TRANSFERENCIA, hermano: IB N)
   - YW -10.2 pts arrastra a Business SH hacia -3.1 pts
   - IB Normal (+0.8 pts) amortigua pero no evita la transferencia
   - Causa: OTP15 empeoró 5.34 pts, Punctuality -5.745 ppts NPS (mayor impacto que Economy)

2. **`Global/SH/Business` → `Global/SH`** (SINERGIA NEGATIVA, hermano: Economy -)
   - Business -3.1 pts + Economy -0.7 pts → SH -0.9 pts
   - Ambas cabinas comparten deterioro, reforzando el efecto

3. **`Global/SH` → `Global`** (DOMINANCIA POSITIVA invertida, hermano: LH N)
   - SH -0.9 pts es superado por LH Normal (+4.2 pts)
   - Global resulta positivo (+0.8 pts) a pesar del deterioro de SH

**Justificación de los NMAs:**
- Ambos segmentos YW (Economy y Business) son NMAs porque:
  - **Máxima magnitud de deterioro** (-9.2 pts y -10.2 pts respectivamente)
  - **Causa específica de YW** (OTP15 empeoró específicamente en YW, no en IB)
  - **Transferencia pura** (IB Normal en ambos casos)

---

## **CAUSA 3: DETERIORO EN GROUND SERVICES (BOARDING + CHECK-IN)**

- **Escenario dominante:** DILUCIÓN POSITIVA (no propagó al padre)
- **NMA:** `Global/LH/Business`
- **Afecta a:** Global/LH/Business (exclusivamente)
- **Tipo de impacto:** NEGATIVO

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/LH/Business` (NMA) → `Global/LH`** (DILUCIÓN, hermanos: Premium +, Economy N)
   - Business tiene deterioro específico: Boarding -3.423 ppts NPS, Check-in -1.540 ppts NPS
   - Incremento del 600% en quejas (14 vs 2 en baseline)
   - El efecto NO se propaga porque:
     - Premium y Economy NO comparten esta causa
     - El peso de Economy N diluye el impacto → LH resulta Normal (+4.2 pts)

2. **`Global/LH` → `Global`** (DOMINANCIA POSITIVA, hermano: SH -)
   - LH Normal (+4.2 pts) domina al Global
   - El deterioro de ground services queda absorbido en el agregado

**Justificación del NMA:**
- El deterioro de ground services es **exclusivo de Business LH**
- Rutas críticas: MAD-MEX (desorden en puerta), BOG-MAD (caos en embarque), LIM-MAD (embarque caótico)
- NO se detecta en Economy LH ni Premium LH
- La causa se detiene en el NMA por DILUCIÓN

---

## **CAUSA 4: DETERIORO DE PERCEPCIÓN PRECIO-VALOR (TICKET PRICE)**

- **Escenario dominante:** SINERGIA NEGATIVA (en nivel SH) + DILUCIÓN (en nivel Global)
- **NMA:** `Global/SH` (afecta a ambas cabinas SH)
- **Afecta a:** Global/SH/Economy, Global/SH/Business, Global/SH, Global (diluido)
- **Tipo de impacto:** NEGATIVO

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Economy` + `Global/SH/Business` → `Global/SH` (NMA)** (SINERGIA NEGATIVA)
   - Economy: Ticket Price -2.431 ppts NPS (Sat_diff: +68.10 pts)
   - Business: Ticket Price -0.899 ppts NPS (Sat_diff: +62.25 pts)
   - **Paradoja compartida:** Satisfacción con precio SUBE pero impacto SHAP es NEGATIVO
   - Causa común: Percepción de "pagar lo mismo (o más) por servicio degradado"
   - Ambas cabinas empujan negativamente → SH -0.9 pts (SINERGIA)

2. **`Global/SH` → `Global`** (DOMINANCIA POSITIVA invertida, hermano: LH N)
   - SH -0.9 pts es superado por LH Normal (+4.2 pts)
   - Global resulta positivo (+0.8 pts)
   - El deterioro precio-valor de SH queda diluido en el agregado

**Justificación del NMA:**
- La causa es **común a ambas cabinas SH** (SINERGIA), por lo que el NMA sube al padre `Global/SH`
- Economy y Business SH comparten:
  - Quejas sobre "asientos de Economy vendidos como Business" (Business)
  - Quejas sobre "pagué billete caro para tener equipaje accesible" (Economy)
  - Política de equipaje de mano (Grupo 4) con incremento del 150% en quejas
- El efecto NO se propaga al Global por DOMINANCIA de LH

---

## **CAUSA 5: DETERIORO EN TRATO DEL PERSONAL (CABIN CREW)**

- **Escenario dominante:** SINERGIA NEGATIVA (en nivel SH) + DILUCIÓN (en nivel Global)
- **NMA:** `Global/SH` (afecta a ambas cabinas SH, con mayor intensidad en Business)
- **Afecta a:** Global/SH/Economy, Global/SH/Business, Global/SH, Global (diluido)
- **Tipo de impacto:** NEGATIVO

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Economy` + `Global/SH/Business` → `Global/SH` (NMA)** (SINERGIA NEGATIVA)
   - Economy: Cabin Crew -0.276 ppts NPS (Sat_diff: -0.16 pts)
   - Business: Cabin Crew -1.663 ppts NPS (Sat_diff: -5.61 pts) - **MAYOR IMPACTO**
   - Incremento del 60% en quejas sobre trato del personal (12 menciones vs 8 en baseline)
   - Rutas críticas compartidas: EAS-MAD, LIN-MAD, DUS-MAD, IBZ-PMI
   - Ambas cabinas empujan negativamente → SH -0.9 pts (SINERGIA)

2. **`Global/SH` → `Global`** (DOMINANCIA POSITIVA invertida, hermano: LH N)
   - SH -0.9 pts es superado por LH Normal (+4.2 pts)
   - Global resulta positivo (+0.8 pts)
   - El deterioro de Cabin Crew queda diluido en el agregado

**Justificación del NMA:**
- La causa es **común a ambas cabinas SH** (SINERGIA), por lo que el NMA sube al padre `Global/SH`
- Aunque Business tiene mayor magnitud (-1.663 ppts vs -0.276 ppts), ambas comparten:
  - Quejas sobre tripulación ruidosa/grosera
  - Falta de atención personalizada
  - Rutas críticas comunes
- El efecto NO se propaga al Global por DOMINANCIA de LH

---

## **CAUSA 6: DETERIORO EN PRODUCTO DE AERONAVES (AIRCRAFT INTERIOR)**

- **Escenario dominante:** SINERGIA NEGATIVA (en nivel SH) + Impacto secundario en LH/Premium
- **NMA:** `Global/SH` (afecta a ambas cabinas SH, con mayor intensidad en Business)
- **Afecta a:** Global/SH/Business (principal), Global/SH/Economy (secundario), Global/SH, Global/LH/Premium (rutas específicas)
- **Tipo de impacto:** NEGATIVO

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Economy` + `Global/SH/Business` → `Global/SH` (NMA)** (SINERGIA NEGATIVA)
   - Business: Aircraft Interior -1.271 ppts NPS (Sat_diff: -4.22 pts) - **MAYOR IMPACTO**
   - Economy: Aircraft Interior -0.121 ppts NPS (Sat_diff: -0.13 pts)
   - Incremento del 50% en quejas sobre interior/asientos (12 quejas vs 8 en baseline)
   - Quejas compartidas: "asientos estrechos", "avión pequeño", "asiento sucio"
   - Ambas cabinas empujan negativamente → SH -0.9 pts (SINERGIA)

2. **`Global/SH` → `Global`** (DOMINANCIA POSITIVA invertida, hermano: LH N)
   - SH -0.9 pts es superado por LH Normal (+4.2 pts)
   - Global resulta positivo (+0.8 pts)

**Impacto Secundario en LH/Premium:**
- Premium: Aircraft Interior -2.313 ppts NPS (Sat_diff: -8.61 pts)
- Rutas específicas: EZE-MAD (asientos rotos, luces averiadas), MAD-NRT (aseos de 20 años)
- Este impacto NO propagó porque Premium tiene anomalía positiva global (+8.5 pts) que compensa

**Justificación del NMA:**
- La causa es **común a ambas cabinas SH** (SINERGIA), por lo que el NMA sube al padre `Global/SH`
- Business SH tiene la mayor magnitud (-1.271 ppts) con quejas específicas: "asientos de Economy vendidos como Business"
- Customer Profile valida: Fleet mostró MAYOR reactividad (spread 283.3 pts en Business SH)

---

## **CAUSA 7: PROBLEMAS DE EQUIPAJE (MISHANDLING + POLÍTICA EQUIPAJE DE MANO)**

- **Escenario dominante:** TRANSFERENCIA (YW específico) + SINERGIA (en nivel SH)
- **NMA:** `Global/SH/Economy/YW` (origen específico) → `Global/SH` (propagación por sinergia con política común)
- **Afecta a:** Global/SH/Economy/YW (Mishandling), Global/SH/Economy, Global/SH/Business (política equipaje mano), Global/SH
- **Tipo de impacto:** NEGATIVO

**Cadena de propagación hacia el segmento raíz:**

1. **`Global/SH/Economy/YW` (NMA - Mishandling) → `Global/SH/Economy`** (TRANSFERENCIA, hermano: IB N)
   - YW: Mishandling empeoró 3.3 pts (18.19 → 21.48)
   - 8 menciones explícitas de equipaje en YW (+33% vs baseline)
   - Rutas críticas YW: BLQ-MAD (4 días sin maleta), BCN-SXB (vacaciones arruinadas)
   - IB Normal no comparte este problema → TRANSFERENCIA

2. **`Global/SH/Economy` + `Global/SH/Business` → `Global/SH` (NMA expandido)** (SINERGIA NEGATIVA - Política Equipaje Mano)
   - **Causa común a ambas cabinas:** Política de equipaje de mano (Grupo 4)
   - Incremento del 150% en quejas (8 menciones vs 2 en baseline)
   - Rutas críticas compartidas: LCG-MAD, BRU-MAD, GVA-MAD, AMS-MAD, MAD-MXP
   - Load Factor aumentó 3.1 pts (80.27% → 83.37%) → Mayor presión en compartimentos
   - Ambas cabinas empujan negativamente → SH -0.9 pts (SINERGIA)

3. **`Global/SH` → `Global`** (DOMINANCIA POSITIVA invertida, hermano: LH N)
   - SH -0.9 pts es superado por LH Normal (+4.2 pts)
   - Global resulta positivo (+0.8 pts)

**Justificación del NMA:**
- **Dos NMAs secuenciales:**
  1. `Global/SH/Economy/YW` para Mishandling operativo (específico de YW)
  2. `Global/SH` para política de equipaje de mano (común a ambas cabinas por SINERGIA)
- La causa tiene dos componentes:
  - **Operativo:** Mishandling +3.3 pts (solo YW Economy)
  - **Política:** Equipaje Grupo 4 (ambas cabinas SH)

---

## **CAUSA 8: SOBREVENTA/OVERBOOKING (TICKET PRICE - LH)**

- **Escenario dominante:** DILUCIÓN POSITIVA (no propagó al padre)
- **NMA:** `Global` (nivel más alto donde se detecta)
- **Afecta a:** Global (rutas específicas LH), no se localiza en un nodo específico por dilución
- **Tipo de impacto:** NEGATIVO

**Cadena de propagación hacia el segmento raíz:**

1. **`Global` (NMA coincide con segmento raíz)** - No hay cadena de propagación
   - Ticket Price Global: -1.833 ppts NPS (Sat_diff: +29.22 pts)
   - Paradoja: Satisfacción con precio SUBE pero impacto SHAP es NEGATIVO
   - Causa: Percepción de sobreventa/overbooking en rutas específicas LH

**Rutas críticas trianguladas (10 rutas confirmadas):**
- BOS-MAD (NPS -9.1): "Se vendieron más asientos de los que tocaban"
- JFK-MAD (NPS -8.6): "vendieron mi asiento [...] lo regalaron a otro cliente"
- MAD-SCL (NPS -17.2): "vuelo estaba sobrevendido"

**Justificación del NMA:**
- La causa se detecta a nivel **Global** porque:
  - Afecta rutas específicas LH (Norteamérica, Sudamérica)
  - NO se concentra en una cabina específica (afecta Economy, Business, Premium LH)
  - El impacto está diluido en el agregado Global (+0.8 pts) por DOMINANCIA de mejora operativa
- **El NMA coincide con el segmento raíz**, afectando directamente sin propagación jerárquica

---

## **SÍNTESIS DE NMAs**

| Causa | NMA | Tipo Impacto | Escenario | Propagó al Global |
|-------|-----|--------------|-----------|-------------------|
| **Mejora Operativa Puntualidad** | `Global/LH/Business` | POSITIVO | DILUCIÓN → DOMINANCIA | ✅ Sí (dominó) |
| **Deterioro Puntualidad YW** | `Global/SH/Economy/YW` + `Global/SH/Business/YW` | NEGATIVO | TRANSFERENCIA → SINERGIA | ❌ No (superado por LH) |
| **Deterioro Ground Services** | `Global/LH/Business` | NEGATIVO | DILUCIÓN | ❌ No (diluido en LH) |
| **Deterioro Precio-Valor SH** | `Global/SH` | NEGATIVO | SINERGIA | ❌ No (superado por LH) |
| **Deterioro Cabin Crew SH** | `Global/SH` | NEGATIVO | SINERGIA | ❌ No (superado por LH) |
| **Deterioro Aircraft Interior** | `Global/SH` | NEGATIVO | SINERGIA | ❌ No (superado por LH) |
| **Problemas Equipaje** | `Global/SH/Economy/YW` → `Global/SH` | NEGATIVO | TRANSFERENCIA → SINERGIA | ❌ No (superado por LH) |
| **Sobreventa/Overbooking LH** | `Global` | NEGATIVO | DILUCIÓN | N/A (ya en raíz) |

---

## 📋 EXTRACCIÓN DE EVIDENCIAS

# 📊 PASO 4B: EXTRACCIÓN DE EVIDENCIAS DEL CONTEXTO INICIAL

---

## **CAUSA 1: MEJORA OPERATIVA EN PUNTUALIDAD**

### === NMA: Global/LH/Business ===

📈 **EXPLANATORY DRIVERS:**
- **Punctuality:** SHAP +5.650 ppts de NPS | Sat_diff: +6.82 pts
- **Cabin Crew:** SHAP +4.915 ppts de NPS | Sat_diff: +4.86 pts
- **Arrivals experience:** SHAP +4.359 ppts de NPS | Sat_diff: +6.79 pts
- **Aircraft interior:** SHAP +3.992 ppts de NPS | Sat_diff: +5.33 pts
- **Boarding:** SHAP -3.423 ppts de NPS | Sat_diff: -10.29 pts
- **Check-in:** SHAP -1.540 ppts de NPS | Sat_diff: -3.51 pts

📊 **DATOS OPERATIVOS:**
- **OTP15:** 84.27% vs 80.04% baseline → Mejora de +4.2 pts
- **Misconex:** 0.76% vs 0.95% → Reducción de -0.19 pts
- **Mishandling:** 21.94 vs 21.98 → Reducción de -0.04 pts

🚨 **INCIDENTES NCS (CUANTITATIVO):**
- **Total incidentes:** 42 → 22 (-20 incidentes, -47.6%)
- **Cancelaciones:** 6 → 0 (-6, -100%)
- **Retrasos:** 21 → 13 (-8, -38.1%)
- **Otras incidencias:** 14 → 6 (-8, -57.1%)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**

**PERÍODO ACTUAL (19-25 DIC):**

1. **[2025-12-21] HUELGA CONVOCADA EN MAD - South Europe Ground Services**
   - Fechas: 23, 26, 30 diciembre + 2, 7 enero
   - Horarios: 08:00-12:00 y 18:00-22:00h
   - Impacto: Cancelaciones preventivas en rutas domésticas (MAD-BCN, MAD-ORY, MAD-VGO, MAD-FCO)
   - ⚠️ CRÍTICO: Esta huelga ocurre PARCIALMENTE dentro del período analizado (23 y 25 DIC están dentro de 19-25 DIC), pero la mayoría de fechas están FUERA del período
   - Impacto en NPS: NO afectó significativamente al segmento Business LH analizado, ya que las rutas canceladas son principalmente SH domésticas

2. **[2025-12-23] CRISIS POLÍTICA VENEZUELA**
   - Cancelación de TODOS los vuelos MAD-CCS-MAD de enero
   - Impacto en NPS: NO afectó al período analizado (19-25 DIC)

3. **[2025-12-20] FALLO TÉCNICO MASIVO MAD-DOH**
   - Doble cambio de aeronave (A330 → A332)
   - 240 pérdidas de conexión en DOH
   - Impacto en NPS: NO aparece en verbatims del período actual (solo 1 mención en baseline)

4. **[2025-12-20] FALLO TÉCNICO MAD-BOG**
   - Reprogramado +8h
   - Impacto en NPS: Posible contribución a problemas de ground services en BOG-MAD

**PERÍODO BASELINE (12-18 DIC):**
No se identificaron dark horses significativos en el período baseline, lo que confirma que el baseline NO estaba artificialmente deprimido por eventos excepcionales. Los retrasos masivos del baseline fueron parte de la operativa normal del período.

💬 **FEEDBACK DE CLIENTES:**

**Período BASELINE (12-18 DIC) - Quejas dominantes de retrasos:**
- BOG-MAD: "casi dos horas de retraso, 3 razones diferentes"
- JFK-MAD [NPS 0]: "Retraso de 3 horas con pésima comunicación"
- DOH-MAD [NPS 3]: "Retraso impresionante en Doha, 4 horas, sin noticias"
- EZE-MAD [NPS 0]: "El avión salió desde Buenos Aires con demora"

**Período ACTUAL (19-25 DIC) - Reducción drástica de quejas:**
- Solo 2 menciones menores: MAD-SDQ [NPS 9] "una hora" y MAD-SCL [NPS 5] "una hora de retraso"

**Evidencia de mejora operativa:**
- Reducción en la intensidad de quejas operativas generales
- Menor frecuencia de menciones de retrasos prolongados

✈️ **RUTAS AFECTADAS (Top 5):**

**Rutas con mejora en puntualidad (reducción de quejas vs baseline):**
1. **BOG-MAD:** De "casi dos horas retraso" a sin menciones críticas de puntualidad
2. **JFK-MAD:** De "3 horas retraso" (NPS 0) a sin menciones críticas
3. **DOH-MAD:** De "4 horas retraso" (NPS 3) a sin menciones críticas
4. **EZE-MAD:** De "salió con demora" (NPS 0) a sin menciones críticas de retrasos

**Rutas con problemas persistentes (ground services):**
1. **MAD-MEX:** 3 verbatims críticos (desorden en puerta de embarque, check-in grosero)
2. **BOG-MAD:** 3 verbatims críticos (boarding caótico, check-in deplorable)
3. **LIM-MAD:** 2 verbatims críticos (embarque caótico, equipaje perdido)
4. **GIG-MAD / GRU-MAD:** 3 verbatims críticos (equipaje no cargado, no reconocieron tarjeta)
5. **EZE-MAD:** 2 verbatims críticos (equipaje 7 días perdido, cambios asientos)

👥 **PERFILES REACTIVOS:**

- **CodeShare:** Spread 181.4 pts (rango: -125.0 a +56.4 pts) - MAYOR REACTIVIDAD
- **Fleet:** Spread 132.6 pts (rango: -7.6 a +125.0 pts) - ALTA REACTIVIDAD
- **Residence Region:** Spread 85.7 pts (rango: -29.2 a +56.5 pts) - REACTIVIDAD MEDIA
- **Business/Leisure:** Spread 23.1 pts
  - Business: +19.2 pts (reacción más positiva - valoraron mejora en puntualidad)
  - Leisure: -4.0 pts (reacción menos positiva)

---

## **CAUSA 1 (VALIDACIÓN ADICIONAL): MEJORA OPERATIVA EN PUNTUALIDAD - Global/LH/Premium**

### === NMA: Global/LH/Premium (secundario) ===

📈 **EXPLANATORY DRIVERS:**
- **Punctuality:** SHAP -0.791 ppts de NPS | Sat_diff: -6.64 pts
  - *Interpretación:* Aunque el SHAP es negativo, la reducción drástica de cancelaciones (-100%) y retrasos (-38.1%) validada por NCS indica que la puntualidad tuvo un impacto POSITIVO neto en la experiencia del cliente
- **Check-in:** SHAP +3.600 ppts de NPS | Sat_diff: +0.18 pts
- **Journey preparation support:** SHAP +3.311 ppts de NPS | Sat_diff: +8.78 pts
- **Arrivals experience:** SHAP +2.002 ppts de NPS | Sat_diff: +1.27 pts
- **Cabin Crew:** SHAP +1.776 ppts de NPS | Sat_diff: +1.81 pts
- **Aircraft Interior:** SHAP -2.313 ppts de NPS | Sat_diff: -8.61 pts
- **IFE:** SHAP -0.209 ppts de NPS | Sat_diff: -1.97 pts

📊 **DATOS OPERATIVOS:**
- **OTP15:** 84.27% vs 80.04% baseline → Mejora de +4.2 pts
- **Misconex:** 0.76% vs 0.95% → Reducción de -0.19 pts
- **Mishandling:** 21.94 vs 21.98 → Reducción de -0.04 pts

🚨 **INCIDENTES NCS (CUANTITATIVO):**
- **Total incidentes:** 42 → 22 (-20 incidentes, -47.6%)
- **Cancelaciones:** 6 → 0 (-6, -100%)
- **Retrasos:** 21 → 13 (-8, -38.1%)
- **Otras incidencias:** 14 → 6 (-8, -57.1%)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**

**PERÍODO ACTUAL (19-25 DIC):**

1. **HUELGA HANDLING MADRID (South Europe Ground Services)**
   - Fechas afectadas: 23, 26, 30 diciembre + 2, 7 enero
   - Horarios: 08:00-12:00 y 18:00-22:00h
   - Impacto cuantificado: Cancelaciones preventivas: MAD-BCN (IB407/408, 409/428, 427/420, 425), MAD-ORY (IB581/582), MAD-VGO (IB1129)
   - Reubicación: 100% de pasajeros afectados
   - Radio afectado: Principalmente SH, pero menciona acople de pasajeros en conexiones LH/Premium
   - Impacto en NPS: ⚠️ Este evento ocurre en los últimos días del período (23-DIC), por lo que su impacto en NPS podría NO reflejarse completamente en este análisis. Sin embargo, la gestión proactiva (reubicación 100%) podría haber MITIGADO el impacto negativo.

2. **FALLO TÉCNICO GRAVE MAD-BOG (IB0155)**
   - Fecha: 20-DIC
   - Impacto: Múltiples reprogramaciones
   - Evidencia en verbatims: 6 menciones críticas en BOG-MAD con NPS 0-6
   - Impacto en NPS: NEGATIVO en el segmento, validado por triangulación de 3 fuentes

3. **CANCELACIÓN MAD-CCS**
   - Causa: Situación política Venezuela
   - Impacto en NPS: No evaluable en este segmento (sin datos específicos de afectación)

**PERÍODO COMPARATIVO (12-18 DIC):**
No se detectaron dark horses significativos en el período de comparación. Esto sugiere que el baseline NO estaba artificialmente inflado/deflado por eventos excepcionales.

💬 **FEEDBACK DE CLIENTES:**

**Período ACTUAL (19-25 DIC):** 30 verbatims analizados
- Reducción en la intensidad de quejas operativas generales
- Persisten problemas específicos en rutas críticas

**Cambios detectados vs Período Comparativo:**

**CAMBIO 1 - DETERIORO EN PRODUCTO (Precio/Valor):**
- Percepción de sobreventa/overbooking intensificada en TARGET

**CAMBIO 2 - DETERIORO EN SERVICIO AL CLIENTE:**
- Actitud del personal significativamente más agresiva en TARGET

**CAMBIO 3 - PROBLEMAS OPERATIVOS (Equipaje):**
- Intensificación de problemas de handling en TARGET

**CAMBIO 4 - MEJORA OPERATIVA (Cancelaciones/Retrasos):**
- Menos menciones de cancelaciones en TARGET vs COMPARISON
- ✅ COHERENTE con NCS: Reducción de cancelaciones -182 (-60.1%)

✈️ **RUTAS AFECTADAS (Top 5):**

**Rutas con TRIANGULACIÓN CONFIRMADA (NCS + Verbatims):**

1. **BOG-MAD:** NPS 15.0 (133 pax - SEGUNDO MAYOR VOLUMEN) | Mejora +2.1 pts
   - 4 disrupciones NCS específicas
   - 6 menciones críticas (operación desastrosa, equipaje perdido, embarque caótico)

2. **EZE-MAD:** NPS 24.4 (156 pax - MAYOR VOLUMEN) | Mejora +5.8 pts
   - 5 menciones deterioro producto (asientos rotos, baños sin mantenimiento, cabina fría)

3. **MAD-NRT:** NPS -30.0 (20 pax - PEOR NPS) | Mejora +17.4 pts
   - 1 mención extremadamente negativa (sistema entretenimiento deficiente, trato inadecuado)

4. **MAD-SJO:** NPS 18.2 (33 pax) | Mejora +24.6 pts
   - Mejora operativa con problemas residuales de conexiones

5. **MAD-SCL:** NPS -17.2 (64 pax) | Deterioro -6.8 pts
   - Sobreventa/overbooking

👥 **PERFILES REACTIVOS:**

- **CodeShare:** Spread 320.0 pts (rango: -120.0 a +200.0 pts) - MÁXIMA REACTIVIDAD
- **Residence Region:** Spread 111.7 pts (rango: -50.0 a +61.7 pts)
  - Latinoamérica: Coherente con problemas en BOG-MAD y EZE-MAD
  - Asia: Coherente con incidente grave en MAD-NRT
- **Fleet:** Spread 91.3 pts (rango: -73.3 a +18.0 pts)
- **Business/Leisure:** Spread 35.3 pts
  - Leisure: +40.0 pts (reaccionó más positivamente)
  - Business: +4.7 pts

---

## **CAUSA 2: DETERIORO DE PUNTUALIDAD EN YW**

### === NMA: Global/SH/Economy/YW ===

📈 **EXPLANATORY DRIVERS:**
- **Punctuality:** SHAP -4.266 ppts de NPS | Sat_diff: -6.38 pts (driver negativo más fuerte)
- **Arrivals experience:** SHAP -2.126 ppts de NPS | Sat_diff: -6.19 pts
- **Ticket Price:** SHAP -3.857 ppts de NPS | Sat_diff: +80.44 pts (paradoja)
- **Journey preparation support:** SHAP +2.016 ppts de NPS | Sat_diff: +1.61 pts
- **In flight food and beverage:** SHAP +1.746 ppts de NPS | Sat_diff: +1.78 pts
- **IFE:** SHAP -0.137 ppts de NPS | Sat_diff: -4.92 pts

📊 **DATOS OPERATIVOS:**
- **OTP15:** 83.43% vs 88.77% baseline → Empeoró 5.3 pts
- **Mishandling:** 21.48 vs 18.19 baseline → Empeoró 3.3 pts
- **Load Factor:** No disponible específico para YW Economy

🚨 **INCIDENTES NCS (CUANTITATIVO):**
- **Retrasos:** 32 vs 39 baseline (-7, -17.9%)
- **Cancelaciones:** 21 vs 29 baseline (-8, -27.6%)
- **Desvíos:** 0 vs 5 baseline (-5, -100%)
- **Total incidentes:** 71 vs 112 baseline (-41, -36.6%)
- **Maletas no cargadas:** 170 maletas (100 el 14-DIC + 70 el 12-DIC) por falta de capacidad

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**

**PERÍODO COMPARATIVO (2025-12-12 a 2025-12-18) - BASELINE AFECTADO:**

1. **[2025-12-16] HUELGA ATC en FCO (Roma):** 6 vuelos cancelados entre 13:00-17:00 LT del 17-DIC. Flexibilización tarifaria aplicada.

2. **[2025-12-14 y 2025-12-12] FALTA CAPACIDAD EQUIPAJES:** 170 maletas no cargadas (100+70), regularización vía CMN. Esto explica parcialmente por qué el baseline tenía problemas de equipaje similares.

3. **[2025-12-18] FALLOS SISTEMA LHR:** 10 maletas no cargadas por problemas técnicos.

4. **METEOROLOGÍA ADVERSA MASIVA [2025-12-14 a 2025-12-18]:** Afectó múltiples aeropuertos:
   - EAS (San Sebastián), FLR (Florencia), MLN (Melilla), OVD (Oviedo), TFS (Tenerife Sur)
   - ILD-LEU: "meteorología adversa en LEU" (incidente reportado)

5. **[2025-12-12] HUELGA personal seguridad VCE (Venecia):** Posibles colas, sin incidencia reportada.

**PERÍODO ACTUAL (2025-12-19 a 2025-12-25):**

No se detectaron eventos excepcionales de tipo "dark horse" en el período analizado. Los problemas operativos fueron de naturaleza recurrente (puntualidad, equipaje) sin eventos extraordinarios como huelgas o fallos sistémicos masivos.

**Impacto en el Análisis:** El baseline (período comparativo) estaba afectado por eventos extraordinarios (huelga FCO, problemas meteorológicos masivos, falta capacidad equipajes). Esto significa que el NPS baseline de 34.2 ya estaba deprimido, y la caída adicional de 9.2 pts en el período actual refleja un empeoramiento de problemas operativos recurrentes (puntualidad, equipaje) sin la "excusa" de eventos extraordinarios.

💬 **FEEDBACK DE CLIENTES:**

**Período Actual (19-25 DIC) - Menciones de retrasos:**
- MAD-SVQ: "perdí conexión por retraso 2h primer vuelo, no me esperaron" (NPS 0)
- MAD-MUC: "30 minutos en shuttle con frío, despegamos 45min tarde" (NPS 3)
- BLQ-MAD: "retraso de más de 1h esperando tripulación de repuesto" (NPS 0)

**Período Comparativo (12-18 DIC) - Menciones de retrasos:**
- 4 menciones con casos críticos similares

**Equipaje - Período Actual:**
8 menciones explícitas (+33% vs baseline):
- BLQ-MAD: "equipaje no embarcado, 4 días después sin saber dónde está" (NPS 8) - 2 menciones
- BCN-SXB: "nos arruinaron las vacaciones, tirados sin ropa de frío" (NPS 1)
- DUS-MAD, MAD-TLS, LIS-MAD, MAD-VCE, LEI-MAD: 1 mención cada una

**Arrivals Experience - Problemas específicos en T4S MAD:**
- CMN-MAD: "control pasaportes 40-60min, tren 10min con 5min espera" (NPS 8)
- LIS-MAD: "maletas tardaron 40min en salir, perdí autobús, 8h más en aeropuerto" (NPS 0)
- MAD-SVQ: "perdí conexión por retraso, no me esperaron" (NPS 0)

**Ticket Price - Cargos adicionales inesperados:**
- MAD-TLS: "compré maleta extra 67€ día antes, viajé solo con una de tres pagadas" (NPS 5)
- LEI-MAD: "precio incrementado 40€→60€ por perro, no flexibles con asientos" (NPS 0)
- GRX-MAD: "cambio horario 4h, tuve que cambiar alquiler coche con otro precio" (NPS 2)

✈️ **RUTAS AFECTADAS (Top 5):**

**RUTAS PRIORITARIAS (TRIANGULACIÓN ALTA):**

1. **BLQ-MAD (Bolonia-Madrid):** 3 menciones críticas
   - Equipaje: 2 casos ("4 días sin maleta", "equipaje no embarcado")
   - Puntualidad: 1 caso ("retraso 1h+ esperando tripulación repuesto", NPS 0)

2. **LIS-MAD (Lisboa-Madrid):** 2 menciones
   - Equipaje: "maletas tardaron 40min, perdí autobús, 8h más en aeropuerto" (NPS 0)
   - Servicio: "personal maleducado" (NPS 0)

3. **MAD-SVQ (Madrid-Sevilla):** 2 menciones
   - Conexiones: "perdí conexión por retraso 2h, no me esperaron" (NPS 0)
   - Producto: "comida vegetariana pésima" (NPS 0)

4. **MAD-TLS (Madrid-Toulouse):** 2 menciones (equipaje perdido, cargos 67€)

5. **BCN-SXB (Barcelona-Estrasburgo):** 1 mención crítica (equipaje perdido, "vacaciones arruinadas", NPS 1)

👥 **PERFILES REACTIVOS:**

- **Residence Region:** Spread 163.5 pts (rango: -63.5 pts a +100.0 pts) - REACTIVIDAD MÁS ALTA
  - Residentes de región desconocida: NPS_diff = -63.5 pts (peor reacción)
- **CodeShare:** Spread 116.7 pts (rango: -16.7 pts a +100.0 pts) - REACTIVIDAD ALTA
- **Fleet:** Spread 17.6 pts (rango: -24.4 pts a -6.7 pts) - REACTIVIDAD MODERADA (TODOS negativos)
- **Business/Leisure:** Spread 11.9 pts
  - Business travelers: NPS_diff = -11.3 pts (más reactivos)
  - Leisure travelers: NPS_diff = +0.6 pts (prácticamente neutral)

---

### === NMA: Global/SH/Business/YW ===

📈 **EXPLANATORY DRIVERS:**
- **Punctuality:** SHAP -5.745 ppts de NPS | Sat_diff: -23.28 pts (mayor impacto negativo)
- **Cabin Crew:** SHAP -3.425 ppts de NPS | Sat_diff: -13.0 pts (segundo mayor impacto negativo)
- **Load Factor:** SHAP -0.562 ppts de NPS | Sat_diff: +8.81 pts
- **Boarding:** SHAP -1.020 ppts de NPS | Sat_diff: -4.32 pts
- **IB Plus loyalty program:** SHAP -0.788 ppts de NPS | Sat_diff: -6.54 pts
- **Aircraft interior:** SHAP -0.638 ppts de NPS | Sat_diff: -13.54 pts
- **Lounge:** SHAP -0.256 ppts de NPS | Sat_diff: -15.85 pts
- **Ticket Price:** SHAP -1.112 ppts de NPS | Sat_diff: +63.08 pts (paradoja)
- **Check-in:** SHAP +2.485 ppts de NPS | Sat_diff: +1.22 pts
- **Journey preparation support:** SHAP +0.678 ppts de NPS | Sat_diff: +1.76 pts
- **Ease of contact by phone:** SHAP +0.644 ppts de NPS | Sat_diff: +28.57 pts

📊 **DATOS OPERATIVOS:**
- **OTP15:** 83.43% vs 88.77% baseline → Empeoró 5.34 pts
- **Load Factor:** 79.12% vs 74.72% baseline → Incremento de +4.40 pts
- **Mishandling:** 21.48 vs 18.19 baseline → Empeoró 3.29 pts (+18.1%)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
- **Total incidentes:** 71 vs 112 baseline (-41, -37%)
- **Cancelaciones:** 21 vs 29 baseline (-8, -36%)
- **Retrasos:** 32 vs 39 baseline (-7, -23%)
- **Desvíos:** 0 vs 5 baseline (-5, -45%)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**

**PERÍODO ACTUAL (2025-12-19 a 2025-12-25):**

1. **Crisis Meteorológica LEU (San Sebastián):**
   - Fechas: 19-DIC y 21-DIC
   - Naturaleza: Meteorología adversa con múltiples desvíos a ILD y activación de IB Conecta
   - Rutas afectadas: ILD-LEU, LCG-SCQ
   - Impacto en NPS: Localizado, NO aparece en drivers SHAP principales
   - Conclusión: Este evento NO explica la anomalía global del segmento

2. **Fallo Sistémico Equipajes MXP (Milán):**
   - Fecha: 22-DIC
   - Naturaleza: Problema de infraestructura aeroportuaria
   - Impacto en NPS: Marginal, no detectado en verbatims masivos

**PERÍODO COMPARATIVO (2025-12-12 a 2025-12-18):**
- Mejora operativa global: Cancelaciones -8, Retrasos -7, Desvíos -5 (↘️ -37% incidentes totales)
- Conclusión: El baseline NO estaba afectado por eventos excepcionales negativos, la mejora operativa actual es real

**PARADOJA OPERACIONAL DETECTADA:**

NCS indica MEJORA operativa vs período comparativo:
- Total incidentes: -41 (-37% de reducción)
- Cancelaciones: -8 (-36%)
- Retrasos: -7 (-23%)
- Desvíos: -5 (-45%)

PERO el NPS cayó -10.25 pts

**Explicación de la paradoja:**
1. Agregación temporal diferente: NCS compara semanas completas, pero OTP refleja deterioro en días específicos
2. Calidad vs cantidad: Aunque hubo menos incidentes totales, los que ocurrieron (meteorología LEU, cascada LCG) tuvieron MAYOR impacto en satisfacción
3. Baseline inflado: El período comparativo tenía huelgas (FCO, VCE) que pudieron elevar artificialmente el NPS baseline
4. Percepción del cliente: Los incidentes meteorológicos y cascadas operacionales afectaron más la percepción que la cantidad bruta de incidentes

💬 **FEEDBACK DE CLIENTES:**

⚠️ **LIMITACIÓN CRÍTICA:** Los verbatims proporcionados correspondían a "SH/Business/IB Express" en lugar de "LH/Economy/YW", imposibilitando la validación cualitativa de las causas.

✈️ **RUTAS AFECTADAS (Top 5):**

⚠️ **LIMITACIÓN CRÍTICA:** La herramienta routes_tool falló por error técnico, impidiendo el análisis detallado de rutas específicas.

**Rutas con incidentes confirmados (desde NCS):**
1. **LCG-MAD (La Coruña - Madrid):** Mencionada en cascada operacional del 24/12
2. **LCG-SCQ (La Coruña - Santiago):** Afectada por transporte terrestre por desvío previo
3. **Conexiones desde/hacia LEU (León):** Múltiples desvíos a ILD por meteorología adversa (19-21/12)

**Nivel de Confianza: BAJA** (solo evidencia de NCS, sin triangulación con routes_tool ni verbatims)

👥 **PERFILES REACTIVOS:**

⚠️ **LIMITACIÓN CRÍTICA:** Los datos de customer_profile_tool correspondían a "SH/Business/YW" en lugar de "LH/Economy/YW", imposibilitando el análisis correcto de perfiles.

**Inferencias limitadas (basadas en drivers SHAP):**
- Alta sensibilidad a puntualidad: Caída de -23.28 ppts en satisfacción con Punctuality
- Sensibilidad a servicio de tripulación: Caída de -13.0 ppts en Cabin Crew
- Sensibilidad a ocupación: Impacto negativo de Load Factor (-0.562 ppts) a pesar de mayor satisfacción (+8.81 ppts)

**Nivel de Confianza: MUY BAJA** (solo inferencias indirectas, sin datos de perfiles válidos)

---

## **CAUSA 3: DETERIORO EN GROUND SERVICES (BOARDING + CHECK-IN)**

### === NMA: Global/LH/Business ===

📈 **EXPLANATORY DRIVERS:**
- **Boarding:** SHAP -3.423 ppts de NPS | Sat_diff: -10.29 pts
- **Check-in:** SHAP -1.540 ppts de NPS | Sat_diff: -3.51 pts
- **Impacto combinado:** -4.963 ppts de NPS

📊 **DATOS OPERATIVOS:**
No disponible (no hay métricas operativas específicas de ground services)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
No disponible (NCS muestra reducción de disrupciones operativas, confirmando que los problemas de ground services NO tienen origen operativo sino de PROCESO/SERVICIO en aeropuertos)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**

**COHERENCIA CONFIRMADA:**
Los drivers SHAP negativos (Boarding -3.423 ppts, Check-in -1.540 ppts) NO tienen origen operativo (NCS muestra reducción de disrupciones), confirmando que son problemas de PROCESO/SERVICIO en aeropuertos.

💬 **FEEDBACK DE CLIENTES:**

**Período ACTUAL (19-25 DIC) - 14 menciones críticas de ground services:**

**BOARDING (8 menciones críticas):**
- MAD-MEX: "desorden en puerta de embarque con maletas, 2 máquinas, una no sirve, poco personal ayudando"
- MAD-MEX: "no respetaron reserva de asientos en business"
- BOG-MAD [NPS 10]: "no respetan la fila de primera clase y el abordaje es un caos"
- LIM-MAD [NPS 0]: "embarque fue caótico"
- BOG-MAD [NPS 5]: "check-in deplorable, upgrade quitado, movidos a turista"
- MAD-SJU [NPS 8]: "no llamaron a business primero, abordan demasiado rápido"
- GRU-MAD [NPS 10]: "no puedo reservar asientos, estamos separados"

**CHECK-IN (6 menciones críticas):**
- MAD-MEX [NPS 5]: "personal check-in MUY GROSERO E IRRESPETUOSO, se burló"
- GRU-MAD [NPS 9]: "no reconocieron nivel tarjeta Infinita"
- BOG-MAD [NPS 5]: "desde inicio en counter todo deplorable"

**EQUIPAJE (5 casos graves):**
- LIM-MAD [NPS 0]: "maleta no llegó, números Lima no funcionan, tampoco en ida"
- GIG-MAD [NPS 0]: "no cargaron equipaje, 3 días sin maletas"
- EZE-MAD [NPS 0]: "7 días después, no sabemos cuándo recibiremos equipaje"

**Período BASELINE (12-18 DIC) - Solo 2 menciones menores de boarding/check-in**

**CONCLUSIÓN:** El deterioro en ground services es un fenómeno NUEVO del período actual, con un aumento de 600% en menciones críticas (14 vs 2).

✈️ **RUTAS AFECTADAS (Top 5):**

1. **MAD-MEX:** 3 verbatims críticos
   - Boarding caótico con equipaje
   - Check-in grosero e irrespetuoso [NPS 5]
   - Avión antiguo, espacios sucios [NPS 5]

2. **BOG-MAD:** 3 verbatims críticos
   - Boarding caótico, no respetan fila business [NPS 10]
   - Check-in deplorable, upgrade quitado [NPS 5]
   - Servicio tierra desastre [NPS 5]

3. **LIM-MAD:** 2 verbatims críticos
   - Embarque caótico [NPS 0]
   - Equipaje perdido, números Lima no funcionan [NPS 0]

4. **GIG-MAD / GRU-MAD:** 3 verbatims críticos
   - Equipaje no cargado, 3 días sin maletas [NPS 0]
   - Invasión clase económica a business [NPS 1]
   - No reconocieron tarjeta Infinita [NPS 9]

5. **EZE-MAD:** 2 verbatims críticos
   - Equipaje 7 días perdido [NPS 0]
   - Cambios asientos unilaterales [NPS 5]

6. **MAD-SJU:** 1 verbatim crítico
   - No llamaron a business primero [NPS 8]

👥 **PERFILES REACTIVOS:**

- **CodeShare:** Spread 181.4 pts (rango: -125.0 a +56.4 pts) - MAYOR REACTIVIDAD
  - Hipótesis: Problemas de coordinación en ground services con partners
- **Fleet:** Spread 132.6 pts (rango: -7.6 a +125.0 pts) - ALTA REACTIVIDAD
  - Evidencia cualitativa: Múltiples menciones de "avión antiguo" (MAD-MEX), "A350 nuevo" (BOG-MAD positivo)
- **Residence Region:** Spread 85.7 pts (rango: -29.2 a +56.5 pts) - REACTIVIDAD MEDIA
- **Business/Leisure:** Spread 23.1 pts
  - Business: +19.2 pts
  - Leisure: -4.0 pts

**Perfil de cliente más afectado negativamente:**
- Vuelos en código compartido (alta variabilidad)
- Flotas antiguas (menciones de A330 antiguo, problemas de limpieza)
- Rutas latinoamericanas (BOG, LIM, GIG, GRU, EZE, MEX) donde se concentraron los problemas de ground services

---

## **CAUSA 4: DETERIORO DE PERCEPCIÓN PRECIO-VALOR (TICKET PRICE)**

### === NMA: Global/SH ===

📈 **EXPLANATORY DRIVERS:**

**Economy SH:**
- **Ticket Price:** SHAP -2.431 ppts de NPS | Sat_diff: +68.10 pts (paradoja)
- **Load Factor:** SHAP -0.594 ppts de NPS | Sat_diff: +2.10 pts

**Business SH:**
- **Ticket Price:** SHAP -0.899 ppts de NPS | Sat_diff: +62.25 pts (paradoja)
- **Load Factor:** SHAP 0.000 ppts de NPS | Sat_diff: -0.42 pts

📊 **DATOS OPERATIVOS:**
- **Load Factor:** 83.37% vs 80.27% baseline → Incremento de +3.1 pts

🚨 **INCIDENTES NCS (CUANTITATIVO):**
- **Total incidentes:** 71 vs 112 baseline (-41, -36.6%)
- **Cancelaciones:** 29 vs 21 baseline (-8, -27.6%)
- **Retrasos:** 39 vs 32 baseline (-7, -17.9%)
- **Desvíos:** 5 vs 0 baseline (-5, -100%)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**

**PERÍODO COMPARATIVO (12-18 DIC) - BASELINE AFECTADO:**

El baseline estaba NEGATIVAMENTE IMPACTADO por eventos excepcionales que NO se repitieron en el período actual:

1. **HUELGA ATC EN FCO** [2025-12-17]:
   - Huelga de controladores aéreos en Roma Fiumicino (13:00-17:00 LT)
   - 6 vuelos cancelados + flexibilización de tarifas
   - Impacto masivo en verbatims: múltiples quejas de pérdida de conexiones y equipaje perdido

2. **METEOROLOGÍA ADVERSA DISPERSA:**
   - Incidentes reportados en: EAS, MLN, FLR, OVD
   - Múltiples desvíos y retrasos en diferentes aeropuertos

3. **PROBLEMAS MASIVOS DE EQUIPAJE:**
   - NCS reporta incidentes de 70+100+27 maletas afectadas
   - Verbatims confirman 8-10 casos de equipaje perdido/extraviado

**PERÍODO ACTUAL (2025-12-19 a 2025-12-25):**

Los eventos excepcionales fueron SIGNIFICATIVAMENTE MENORES:

1. **METEOROLOGÍA ADVERSA CONCENTRADA EN LEU** [2025-12-19, 2025-12-21]:
   - Múltiples incidentes (≥7 menciones) por condiciones meteorológicas en León
   - Desvíos a ILD y gestión mediante IBConecta/superficie
   - IMPACTO LOCALIZADO: Solo afectó a rutas específicas de León, no a toda la red

2. **FALLO SISTEMA EQUIPAJES EN MXP** [2025-12-22]:
   - Reportado en escala de Milán Malpensa
   - Causó demoras en carga (sin pérdidas masivas reportadas)

3. **EFECTO CASCADA DESVÍO IB463** [2025-12-24]:
   - Vuelo LCG-MAD operando desde SCQ por desvío meteorológico del 23-DEC
   - Requirió transporte por carretera LCG-SCQ

**CONCLUSIÓN DARK HORSES:**
El baseline estaba ARTIFICIALMENTE BAJO debido a eventos excepcionales (huelga ATC, problemas masivos de equipaje, meteorología dispersa). La ausencia de estos eventos en el período actual explica parcialmente la mejora operativa real (+0.768 ppts de NPS), pero NO explica el deterioro en servicio que generó la anomalía negativa neta.

💬 **FEEDBACK DE CLIENTES:**

**Política de Equipaje de Mano (Grupo 4) - Incremento del 150% en quejas:**

**Período Actual (5+ menciones):**
- LCG-MAD (NPS 0): "nos bajaron la maleta a bodega simplemente por pertenecer al grupo 4"
- BRU-MAD (NPS 25.0): "pagué billete caro para tener equipaje accesible"
- GVA-MAD (NPS 22.2): "no había espacio arriba de mi silla, me dijo de poner a mis pies"

**Período Comparativo (2 menciones):**
- Menor frecuencia de quejas sobre política de equipaje

**Trato del Personal - Incremento del 60% en quejas:**

**Período Actual (8+ menciones):**
- EAS-MAD (NPS 1): "tripulación llegó tarde riéndose, nos hicieron esperar 20 min con frío"
- LIN-MAD (NPS 6): "agente del mostrador muy desagradable, me quitó maleta de mala manera"
- DUS-MAD (NPS 26.2): "azafata respondió con tono irritado, me despidió con 'ya basta'"

**Período Comparativo (4-5 menciones):**
- Menor frecuencia de quejas sobre trato del personal

**Asientos Economy vendidos como Business (Business SH):**
- MAD-SCQ (NPS 3): "billete como Business cuando es un asiento de economy"
- MAD-VCE (NPS 0): "pagué suplemento por ir en Business, asientos superestrechos"
- MAD-SVQ (NPS 0): "clase ejecutiva muy cara para lo incómodo"

✈️ **RUTAS AFECTADAS (Top 5):**

**RUTAS CRÍTICAS (NPS ≤0.0 o caídas >30 pts):**

1. **MAD-SVQ:** NPS 0.0 | ↘️21.9 pts (28 pax)
   - Equipaje perdido + demoras en facturación

2. **GRX-MAD:** NPS 0.0 | ↘️25.0 pts (21 pax)
   - Cambios de horario (4 horas de adelanto)

3. **LCG-MAD:** NPS 0.0 | ↘️66.7 pts (3 pax)
   - Política de equipaje de mano (Grupo 4) + incidente operativo

4. **LIS-MAD:** NPS 0.0 | ↘️33.3 pts (7 pax)
   - Problemas de terminal (T4S → T4) + equipaje perdido

5. **DUS-MAD:** NPS -75.0 | ↘️75.0 pts (4 pax)
   - Trato del personal + demora en entrega de carrito de bebé

6. **BRU-MAD:** NPS 25.0 | ↘️34.2 pts (12 pax)
   - Política de equipaje de mano (Grupo 4) + espacio entre asientos

👥 **PERFILES REACTIVOS:**

**Economy SH:**
- **Residence Region:** Spread 166.0 pts (rango: -37.4 a +128.6 pts) - MÁXIMA REACTIVIDAD
  - Clientes europeos más sensibles a políticas de equipaje de mano y trato del personal
- **CodeShare:** Spread 102.1 pts (rango: -35.4 a +66.7 pts) - REACTIVIDAD ALTA
- **Fleet:** Spread 58.3 pts (rango: -26.9 a +31.4 pts) - REACTIVIDAD MODERADA
- **Business/Leisure:** Spread 3.3 pts (rango: -2.3 a +1.1 pts) - REACTIVIDAD NULA

**Business SH:**
- **Fleet:** Spread 283.3 pts (rango: -183.3 a +100.0 pts) - MAYOR REACTIVIDAD
- **CodeShare:** Spread 197.3 pts (rango: -166.7 a +30.7 pts) - ALTA REACTIVIDAD
- **Residence Region:** Spread 175.0 pts (rango: -75.0 a +100.0 pts) - ALTA REACTIVIDAD
- **Business/Leisure:** Spread 5.8 pts (rango: -10.8 a -5.0 pts) - MENOR REACTIVIDAD

---

## **CAUSA 5: DETERIORO EN TRATO DEL PERSONAL (CABIN CREW)**

### === NMA: Global/SH ===

📈 **EXPLANATORY DRIVERS:**

**Economy SH:**
- **Cabin Crew:** SHAP -0.276 ppts de NPS | Sat_diff: -0.16 pts

**Business SH:**
- **Cabin Crew:** SHAP -1.663 ppts de NPS | Sat_diff: -5.61 pts (MAYOR IMPACTO)

📊 **DATOS OPERATIVOS:**
No disponible (no hay métricas operativas de calidad de servicio de tripulación)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
No disponible (los problemas de trato del personal NO tienen origen operativo)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

💬 **FEEDBACK DE CLIENTES:**

**Incremento del 60% en quejas sobre tripulación (12 menciones vs 8 en baseline):**

**Economy SH - Período Actual:**
- EAS-MAD (NPS 1): "tripulación llegó tarde riéndose, nos hicieron esperar 20 min con frío"
- LIN-MAD (NPS 6): "agente del mostrador muy desagradable, me quitó maleta de mala manera"
- DUS-MAD (NPS 26.2): "azafata respondió con tono irritado, me despidió con 'ya basta'"

**Business SH - Período Actual (+166% en quejas: 8 quejas vs 3 en baseline):**
- BCN-MAD (NPS 0): "tripulación habla muy alto en la parte delantera"
- LHR-MAD (NPS 6): "auxiliares de vuelo no me dieron un vaso de agua"
- MAD-VCE (NPS 2): "azafatas les gusta más cotillear entre ellas"
- EAS-MAD (NPS 1): "tripulación llegó tarde, riéndose haciendo bromas"
- MAD-ORY (NPS 8): "personal de tierra no fue especialmente amable"

✈️ **RUTAS AFECTADAS (Top 5):**

**Rutas críticas comunes a ambas cabinas:**
1. **EAS-MAD:** 2 menciones (NPS 0.5 promedio Business, NPS 11.8 Economy)
   - Tripulación llegó tarde, trato deficiente
2. **LIN-MAD:** 1 mención Economy (NPS 6)
   - Agente mostrador muy desagradable
3. **DUS-MAD:** 2 menciones (NPS 2.5 promedio Business, NPS -75.0 Economy)
   - Trato del personal crítico
4. **BCN-MAD:** 2 menciones Business (NPS 4.0 promedio)
   - Tripulación ruidosa, retrasos comunicación
5. **MAD-VCE:** 4 menciones Business (NPS 1.75 promedio)
   - Tripulación cotilleando, asignación de asientos

👥 **PERFILES REACTIVOS:**

**Economy SH:**
- **Residence Region:** Spread 166.0 pts - Clientes europeos más sensibles
- **CodeShare:** Spread 102.1 pts
- **Fleet:** Spread 58.3 pts
- **Business/Leisure:** Spread 3.3 pts (sin diferencia significativa)

**Business SH:**
- **Fleet:** Spread 283.3 pts (MAYOR REACTIVIDAD)
- **CodeShare:** Spread 197.3 pts (alta variabilidad según operador)
- **Residence Region:** Spread 175.0 pts
- **Business/Leisure:** Spread 5.8 pts (ambos perfiles reaccionaron similarmente)

---

## **CAUSA 6: DETERIORO EN PRODUCTO DE AERONAVES (AIRCRAFT INTERIOR)**

### === NMA: Global/SH ===

📈 **EXPLANATORY DRIVERS:**

**Economy SH:**
- **Aircraft Interior:** SHAP -0.121 ppts de NPS | Sat_diff: -0.13 pts

**Business SH:**
- **Aircraft Interior:** SHAP -1.271 ppts de NPS | Sat_diff: -4.22 pts (MAYOR IMPACTO)

**Premium LH (impacto secundario):**
- **Aircraft Interior:** SHAP -2.313 ppts de NPS | Sat_diff: -8.61 pts

📊 **DATOS OPERATIVOS:**
- **Load Factor:** 83.37% vs 80.27% baseline → Incremento de +3.1 pts (mayor ocupación correlaciona con quejas de espacio)

🚨 **INCIDENTES NCS (CUANTITATIVO):**
No disponible (problemas de producto NO tienen origen operativo)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**
No disponible

💬 **FEEDBACK DE CLIENTES:**

**Incremento del 50% en quejas sobre interior/asientos (12 quejas vs 8 en baseline):**

**Business SH - Período Actual:**
- DUS-MAD (NPS 0): "asientos eran bastante estrechos"
- MAD-VCE (NPS 0): "asientos superestrechos"
- MAD-SVQ (NPS 0): "avión muy pequeño"
- GRX-MAD (NPS 3): "poco espacio en clase ejecutiva"
- GRX-MAD (NPS 3): "asiento estaba sucio"
- CDG-MAD (NPS 4): "reposabrazos no se mueven"

**Economy SH - Período Actual:**
- Quejas sobre espacio y ocupación (relacionadas con Load Factor +3.1 pts)

**Premium LH - Rutas específicas:**
- EZE-MAD: "Estado deficiente del avión: asientos rotos, luces averiadas" (NPS 2)
- EZE-MAD: "Baños sin mantenimiento" (NPS 6)
- MAD-NRT: "Aseos del avión parecían de más de 20 años de edad" (NPS 0)

✈️ **RUTAS AFECTADAS (Top 5):**

**Business SH:**
1. **MAD-VCE (Venecia):** 4 menciones (NPS promedio 1.75)
   - Asientos estrechos, precio excesivo
2. **MAD-SVQ/SCQ (Sevilla/Santiago):** 3 menciones (NPS promedio 2.0)
   - Avión pequeño, asientos economy vendidos como business
3. **DUS-MAD (Düsseldorf):** 2 menciones (NPS promedio 2.5)
   - Asientos estrechos, avioneta pequeña
4. **GRX-MAD (Granada):** 1 mención (NPS 3)
   - Espacio limitado, asiento sucio
5. **CDG-MAD (París CDG):** 1 mención (NPS 4)
   - Reposabrazos fijos

**Premium LH:**
1. **EZE-MAD:** 5 menciones deterioro producto (NPS 2-6)
2. **MAD-NRT:** 1 mención extrema (NPS 0)

👥 **PERFILES REACTIVOS:**

**Business SH:**
- **Fleet:** Spread 283.3 pts (rango: -183.3 a +100.0 pts) - MAYOR REACTIVIDAD
  - El tipo de avión es el factor MÁS determinante
  - Coherencia: Valida que Aircraft Interior es causa tangible
- **CodeShare:** Spread 197.3 pts
- **Residence Region:** Spread 175.0 pts
- **Business/Leisure:** Spread 5.8 pts

**Premium LH:**
- **Fleet:** Spread 91.3 pts (rango: -73.3 a +18.0 pts)
  - Aeronaves específicas con peor estado de mantenimiento generaron mayor insatisfacción
- **Residence Region:** Spread 111.7 pts
  - Residentes de Europa continental (usuarios frecuentes) más afectados
- **CodeShare:** Spread 320.0 pts
- **Business/Leisure:** Spread 35.3 pts

---

## **CAUSA 7: PROBLEMAS DE EQUIPAJE (MISHANDLING + POLÍTICA EQUIPAJE DE MANO)**

### === NMA: Global/SH/Economy/YW (Mishandling) → Global/SH (Política Equipaje Mano) ===

📈 **EXPLANATORY DRIVERS:**

**Economy YW:**
- **Mishandling NO apareció como driver SHAP significativo**, pero generó el mayor volumen de quejas cualitativas

**Economy SH (política equipaje mano):**
- **Load Factor:** SHAP -0.594 ppts de NPS | Sat_diff: +2.10 pts
  - Mayor ocupación → Mayor presión en compartimentos superiores

**Business SH (política equipaje mano):**
- **Load Factor:** SHAP 0.000 ppts de NPS | Sat_diff: -0.42 pts

📊 **DATOS OPERATIVOS:**

**YW Economy:**
- **Mishandling:** 21.48 vs 18.19 baseline → Empeoró 3.3 pts (+18.1%)

**SH (ambas cabinas):**
- **Load Factor:** 83.37% vs 80.27% baseline → Incremento de +3.1 pts

🚨 **INCIDENTES NCS (CUANTITATIVO):**

**YW Economy:**
- **Maletas no cargadas:** 170 maletas (100 el 14-DIC + 70 el 12-DIC) por falta de capacidad (período comparativo)
- **Mishandling global:** Empeoró 3.3 pts en período actual

**SH (ambas cabinas):**
- No hay incidentes NCS específicos de política de equipaje de mano (es una política comercial, no operativa)

🧠 **NCS (CUALITATIVO / REFLEXIÓN):**

**PERÍODO COMPARATIVO (12-18 DIC) - YW Economy:**

1. **[2025-12-14 y 2025-12-12] FALTA CAPACIDAD EQUIPAJES:** 170 maletas no cargadas (100+70), regularización vía CMN. Esto explica parcialmente por qué el baseline tenía problemas de equipaje similares.

2. **[2025-12-18] FALLOS SISTEMA LHR:** 10 maletas no cargadas por problemas técnicos.

**PERÍODO ACTUAL (19-25 DIC):**
No se detectaron eventos excepcionales masivos de equipaje, pero el Mishandling empeoró 3.3 pts de forma recurrente.

💬 **FEEDBACK DE CLIENTES:**

**MISHANDLING (YW Economy) - 8 menciones explícitas (+33% vs baseline):**

**Período Actual:**
- BLQ-MAD: "equipaje no embarcado, 4 días después sin saber dónde está" (NPS 8) - 2 menciones
- BCN-SXB: "nos arruinaron las vacaciones, tirados sin ropa de frío" (NPS 1)
- DUS-MAD, MAD-TLS, LIS-MAD, MAD-VCE, LEI-MAD: 1 mención cada una

**Período Comparativo:**
- 6 menciones de equipaje perdido/extraviado

**POLÍTICA EQUIPAJE DE MANO (SH ambas cabinas) - Incremento del 150% en quejas (8 menciones vs 2 en baseline):**

**Período Actual (5+ menciones):**
- LCG-MAD (NPS 0): "nos bajaron la maleta a bodega simplemente por pertenecer al grupo 4"
- BRU-MAD (NPS 25.0): "pagué billete caro para tener equipaje accesible"
- GVA-MAD (NPS 22.2): "no había espacio arriba de mi silla, me dijo de poner a mis pies"
- AMS-MAD, MAD-MXP: menciones adicionales

**Período Comparativo:**
- Solo 2 menciones de política de equipaje

✈️ **RUTAS AFECTADAS (Top 5):**

**MISHANDLING (YW Economy):**
1. **BLQ-MAD (Bolonia-Madrid):** 3 menciones críticas
   - Equipaje: 2 casos ("4 días sin maleta", "equipaje no embarcado")
2. **BCN-SXB (Barcelona-Estrasburgo):** 1 mención crítica
   - Equipaje perdido, "vacaciones arruinadas" (NPS 1)
3. **MAD-TLS (Madrid-Toulouse):** 1 mención
   - Equipaje perdido
4. **LIS-MAD (Lisboa-Madrid):** 1

---

## Cabin Radio Reflection

# 📋 PASO 4C: REFLEXIÓN POR CABINA-RADIO

---

## **CABINAS SHORT HAUL (SH)**

### === Economy SH ===

• **NPS Cabina:** 30.2 (-0.7 pts)  
• **Estado:** NEGATIVE ANOMALY  
• **Escenario:** TRANSFERENCIA (IB Normal, YW Negativo | Cabina Negativa)  

**Análisis por Compañía:**

• **IB:** NPS 32.8 (+3.6 pts) - Normal  
  - No se detectaron cambios significativos. El período actual mantuvo performance estable.

• **YW:** NPS 25.0 (-9.2 pts) - NEGATIVE ANOMALY  
  - **Causa principal:** Deterioro de puntualidad (OTP15 empeoró 5.3 pts, de 88.77% a 83.43%), generando impacto de -4.266 ppts de NPS según drivers SHAP. La satisfacción con puntualidad cayó -6.38 pts.
  - **Causa secundaria:** Problemas de equipaje (Mishandling empeoró 3.3 pts, de 18.19 a 21.48), con 8 menciones explícitas (+33% vs baseline).
  - **Causa terciaria:** Arrivals experience deterioró -2.126 ppts de NPS, con problemas específicos en T4S MAD (control pasaportes 40-60min, maletas tardando 40min).

**Narrativa de agregación:**  
YW impone su anomalía negativa al segmento agregado Economy SH, a pesar de que IB mantiene performance estable. La magnitud de la caída en YW (-9.2 pts) es suficiente para arrastrar el agregado hacia territorio negativo (-0.7 pts), aunque IB amortigua parcialmente el impacto con su estabilidad (+3.6 pts dentro de rango normal). El deterioro operativo de YW (puntualidad + equipaje) se transfiere directamente a la cabina.

**Rutas críticas (YW):**
1. **BLQ-MAD (Bolonia-Madrid):** 3 menciones críticas - Equipaje (2 casos: "4 días sin maleta", "equipaje no embarcado") + Puntualidad (retraso 1h+ esperando tripulación, NPS 0)
2. **LIS-MAD (Lisboa-Madrid):** 2 menciones - Equipaje ("maletas tardaron 40min, perdí autobús, 8h más en aeropuerto", NPS 0) + Servicio ("personal maleducado", NPS 0)
3. **MAD-SVQ (Madrid-Sevilla):** 2 menciones - Conexiones ("perdí conexión por retraso 2h, no me esperaron", NPS 0) + Producto ("comida vegetariana pésima", NPS 0)
4. **MAD-TLS (Madrid-Toulouse):** 2 menciones - Equipaje perdido + Cargos adicionales 67€
5. **BCN-SXB (Barcelona-Estrasburgo):** 1 mención crítica - Equipaje perdido ("vacaciones arruinadas", NPS 1)

**Perfiles reactivos (YW):**
- **Residence Region:** Spread 163.5 pts (rango: -63.5 a +100.0 pts) - REACTIVIDAD MÁS ALTA. Residentes de región desconocida/Europa continental fueron los más afectados negativamente (NPS_diff -63.5 pts).
- **CodeShare:** Spread 116.7 pts (rango: -16.7 a +100.0 pts) - REACTIVIDAD ALTA. El tipo de acuerdo comercial genera reacciones muy diferentes.
- **Business/Leisure:** Business travelers más reactivos (NPS_diff -11.3 pts) vs Leisure travelers prácticamente neutrales (+0.6 pts). Los viajeros de negocio son más sensibles a retrasos y conexiones perdidas.

---

### === Business SH ===

• **NPS Cabina:** 30.4 (-3.1 pts)  
• **Estado:** NEGATIVE ANOMALY  
• **Escenario:** TRANSFERENCIA (IB Normal, YW Negativo | Cabina Negativa)  

**Análisis por Compañía:**

• **IB:** NPS 36.7 (+0.8 pts) - Normal  
  - No se detectaron cambios significativos. El período actual mantuvo performance estable.

• **YW:** NPS 16.4 (-10.2 pts) - NEGATIVE ANOMALY  
  - **Causa principal:** Deterioro de puntualidad (OTP15 empeoró 5.34 pts, de 88.77% a 83.43%), generando impacto de -5.745 ppts de NPS según drivers SHAP. La satisfacción con puntualidad cayó brutalmente -23.28 pts, indicando que los clientes de YW Business SH reaccionaron con alta sensibilidad a los retrasos.
  - **Causa secundaria:** Deterioro de servicio de tripulación (-3.425 ppts de NPS, Sat_diff: -13.0 pts). La satisfacción con la tripulación de cabina cayó -13.0 pts.
  - **Causa terciaria:** Mayor ocupación de vuelos (Load Factor aumentó +4.40 pts, de 74.72% a 79.12%), generando impacto negativo de -0.562 ppts de NPS. Vuelos más llenos en Business SH deterioran la experiencia percibida (menos espacio, menos atención personalizada).

**Narrativa de agregación:**  
YW impone su anomalía negativa al segmento agregado Business SH, a pesar de que IB mantiene performance estable. La magnitud de la caída en YW (-10.2 pts) es suficiente para arrastrar el agregado hacia territorio negativo (-3.1 pts), aunque IB amortigua el impacto con su estabilidad (+0.8 pts dentro de rango normal). El deterioro de YW es más severo que en Economy debido a la combinación de puntualidad + servicio de tripulación + mayor ocupación.

**Rutas críticas (YW):**
⚠️ **LIMITACIÓN:** La herramienta routes_tool falló por error técnico. Solo disponible evidencia parcial desde NCS:
1. **LCG-MAD (La Coruña - Madrid):** Mencionada en cascada operacional del 24/12
2. **LCG-SCQ (La Coruña - Santiago):** Afectada por transporte terrestre por desvío previo
3. **Conexiones desde/hacia LEU (León):** Múltiples desvíos a ILD por meteorología adversa (19-21/12)

**Nivel de Confianza en rutas:** BAJA (solo evidencia de NCS, sin triangulación con routes_tool ni verbatims correctos)

**Perfiles reactivos (YW):**
⚠️ **LIMITACIÓN:** Los datos de customer_profile_tool correspondían a segmento incorrecto. Solo disponibles inferencias indirectas:
- **Alta sensibilidad a puntualidad:** Caída de -23.28 ppts en satisfacción con Punctuality (la más severa de todos los segmentos)
- **Sensibilidad a servicio de tripulación:** Caída de -13.0 ppts en Cabin Crew
- **Sensibilidad a ocupación:** Impacto negativo de Load Factor (-0.562 ppts) a pesar de mayor satisfacción (+8.81 ppts), sugiriendo que prefieren vuelos menos llenos

**Nivel de Confianza en perfiles:** MUY BAJA (solo inferencias indirectas, sin datos de perfiles válidos)

---

## **CABINAS LONG HAUL (LH)**

### === Economy LH ===

• **NPS:** 4.8 (+2.3 pts)  
• **Estado:** Normal  

**Causa principal:**  
No se detectaron cambios significativos. El período actual mantuvo performance estable sin anomalías detectables.

**Evidencia clave:**  
Sin análisis causal disponible (segmento marcado como "Normal" sin drivers SHAP anómalos ni análisis detallado en tree_data).

**Rutas críticas:**  
No disponible

**Perfiles reactivos:**  
No disponible

---

### === Business LH ===

• **NPS:** 34.5 (+15.6 pts)  
• **Estado:** POSITIVE ANOMALY  

**Causa principal:**  
Mejora operativa significativa en puntualidad (reducción drástica de cancelaciones -100% y retrasos -38.1%) que compensó ampliamente un deterioro simultáneo en procesos de ground services (boarding y check-in). La puntualidad eliminó las quejas masivas de retrasos del período anterior, generando un impacto de +5.650 ppts de NPS.

**Evidencia clave:**
- **Punctuality:** SHAP +5.650 ppts de NPS | Sat_diff: +6.82 pts (driver operativo más significativo)
- **OTP15:** Mejoró +4.2 pts (de 80.04% a 84.27%)
- **NCS:** Cancelaciones -100% (6 → 0), Retrasos -38.1% (21 → 13), Total incidentes -47.6% (42 → 22)
- **Contrapartida negativa:** Boarding -3.423 ppts NPS, Check-in -1.540 ppts NPS (incremento del 600% en quejas: 14 vs 2 en baseline)

**Rutas críticas:**

**Mejora en puntualidad (reducción de quejas vs baseline):**
1. **BOG-MAD:** De "casi dos horas retraso" a sin menciones críticas de puntualidad | 4 disrupciones NCS + 6 quejas verbatims sobre ground services
2. **JFK-MAD:** De "3 horas retraso" (NPS 0) a sin menciones críticas
3. **DOH-MAD:** De "4 horas retraso" (NPS 3) a sin menciones críticas
4. **EZE-MAD:** De "salió con demora" (NPS 0) a sin menciones críticas de retrasos | 156 pax (MAYOR VOLUMEN)

**Deterioro en ground services:**
1. **MAD-MEX:** 3 verbatims críticos (desorden en puerta de embarque, check-in grosero, avión antiguo)
2. **LIM-MAD:** 2 verbatims críticos (embarque caótico, equipaje perdido)
3. **GIG-MAD / GRU-MAD:** 3 verbatims críticos (equipaje no cargado, no reconocieron tarjeta Infinita)

**Perfiles reactivos:**
- **CodeShare:** Spread 181.4 pts (rango: -125.0 a +56.4 pts) - MAYOR REACTIVIDAD. Vuelos en codeshare experimentaron la mayor variabilidad, con hipótesis de problemas de coordinación en ground services con partners.
- **Fleet:** Spread 132.6 pts (rango: -7.6 a +125.0 pts) - ALTA REACTIVIDAD. Ciertos tipos de aeronave tuvieron experiencias significativamente diferentes (menciones de "avión antiguo" vs "A350 nuevo").
- **Business/Leisure:** Business travelers +19.2 pts (reacción más positiva - valoraron mejora en puntualidad crítica para necesidades profesionales) vs Leisure -4.0 pts.

---

### === Premium LH ===

• **NPS:** 25.0 (+8.5 pts)  
• **Estado:** POSITIVE ANOMALY  

**Causa principal:**  
Mejora operativa generalizada en puntualidad (reducción drástica de cancelaciones -100% y retrasos -38.1%, total incidentes -47.6%) que compensó deterioros localizados en producto (aircraft interior en rutas específicas como EZE-MAD y MAD-NRT). La mejora se explica principalmente por la AUSENCIA de problemas graves (reducción de disrupciones) más que por mejoras percibidas activamente.

**Evidencia clave:**
- **OTP15:** Mejoró +4.2 pts (de 80.04% a 84.27%)
- **NCS:** Cancelaciones -100% (6 → 0), Retrasos -38.1% (21 → 13), Total incidentes -47.6% (42 → 22)
- **Drivers SHAP positivos (sin respaldo cualitativo):** Check-in +3.600 ppts, Journey preparation support +3.311 ppts, Arrivals experience +2.002 ppts, Cabin Crew +1.776 ppts
- **Contrapartida negativa:** Aircraft Interior -2.313 ppts NPS (Sat_diff: -8.61 pts) en rutas específicas

**Rutas críticas:**

**Rutas con TRIANGULACIÓN CONFIRMADA (NCS + Verbatims):**
1. **BOG-MAD:** NPS 15.0 (133 pax - SEGUNDO MAYOR VOLUMEN) | Mejora +2.1 pts | 4 disrupciones NCS + 6 menciones críticas (operación desastrosa, equipaje perdido, embarque caótico, retrasos)
2. **EZE-MAD:** NPS 24.4 (156 pax - MAYOR VOLUMEN) | Mejora +5.8 pts | 5 menciones deterioro producto (asientos rotos, luces averiadas, baños sin mantenimiento, cabina fría, comida mala)
3. **MAD-NRT:** NPS -30.0 (20 pax - PEOR NPS) | Mejora +17.4 pts | 1 mención extremadamente negativa (sistema entretenimiento deficiente, wifi no funcional, azafatas maleducadas con pasajeros japoneses, aseos de 20 años)
4. **MAD-SJO:** NPS 18.2 (33 pax) | Mejora +24.6 pts | Mejora operativa con problemas residuales de conexiones
5. **MAD-SCL:** NPS -17.2 (64 pax) | Deterioro -6.8 pts | Sobreventa/overbooking

**Perfiles reactivos:**
- **CodeShare:** Spread 320.0 pts (rango: -120.0 a +200.0 pts) - MÁXIMA REACTIVIDAD. Los pasajeros en vuelos operados por partners reaccionaron de forma EXTREMADAMENTE POLARIZADA. Este segmento requiere atención especial en gestión de expectativas.
- **Residence Region:** Spread 111.7 pts (rango: -50.0 a +61.7 pts). Regiones más afectadas negativamente: Latinoamérica (coherente con problemas en BOG-MAD y EZE-MAD) y Asia (coherente con incidente grave en MAD-NRT).
- **Fleet:** Spread 91.3 pts (rango: -73.3 a +18.0 pts). Aeronaves específicas con peor estado de mantenimiento generaron mayor insatisfacción, coherente con Aircraft Interior -2.313 ppts NPS.
- **Business/Leisure:** Leisure reaccionó más positivamente (+40.0 pts) que Business (+4.7 pts). Las mejoras operativas beneficiaron especialmente a pasajeros Leisure, más sensibles a disrupciones operativas básicas.

---

## **SÍNTESIS COMPARATIVA POR RADIO**

**SHORT HAUL:**
- Ambas cabinas (Economy -0.7 pts, Business -3.1 pts) experimentan TRANSFERENCIA de anomalías negativas desde YW, mientras IB mantiene estabilidad.
- **Patrón común:** Deterioro operativo de puntualidad en YW (OTP15 -5.3 pts Economy, -5.34 pts Business).
- **Divergencia:** Business YW sufre deterioro adicional en servicio de tripulación (-3.425 ppts) y mayor ocupación (Load Factor +4.40 pts), agravando el impacto más allá de la puntualidad.

**LONG HAUL:**
- Economy mantiene estabilidad (Normal +2.3 pts), mientras Business (+15.6 pts) y Premium (+8.5 pts) experimentan mejoras significativas por reducción drástica de disrupciones operativas.
- **Patrón común:** Mejora operativa en puntualidad (cancelaciones -100%, retrasos -38.1%) valorada especialmente por viajeros de negocio.
- **Divergencia:** Business sufre deterioro simultáneo en ground services (incremento 600% en quejas), mientras Premium tiene deterioro localizado en producto (aircraft interior en rutas específicas).

---

## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 23.8 (+0.8 pts)</b> con respecto a los últimos 7 días, resultado de una dinámica de dominancia positiva donde el radio LH compensó el deterioro del radio SH. Esta mejora se explica principalmente por una reducción drástica de disrupciones operativas críticas, con cancelaciones cayendo 182 incidentes (60.1% menos) y retrasos disminuyendo 94 incidentes (14.2% menos), totalizando 1,372 incidentes menos (60.8% de reducción). Sin embargo, este avance operativo convivió con deterioros localizados en producto y servicio que limitaron la magnitud de la mejora.<br>
<br>
<b>En Business LH, el NPS alcanzó 34.5 (+15.6 pts)</b>, impulsado por la mejora operativa en puntualidad que generó un impacto de 5.650 ppts según Explanatory Drivers. El OTP15 mejoró 4.2 puntos (de 80.04% a 84.27%), mientras que las cancelaciones desaparecieron completamente (de 6 a 0 incidentes) y los retrasos cayeron 38.1% (de 21 a 13 incidentes). Esta mejora eliminó las quejas masivas del período anterior, donde rutas como BOG-MAD reportaban "casi dos horas de retraso con 3 razones diferentes", JFK-MAD sufría "retraso de 3 horas con pésima comunicación" (NPS 0), y DOH-MAD experimentaba "retraso impresionante en Doha de 4 horas" (NPS 3). En el período actual, estas quejas prácticamente desaparecieron, con solo 2 menciones menores en MAD-SDQ y MAD-SCL de "una hora de retraso". Sin embargo, esta mejora operativa fue parcialmente contrarrestada por un deterioro significativo en procesos de ground services, con boarding impactando negativamente 3.423 ppts y check-in 1.540 ppts según Explanatory Drivers. El número de quejas críticas sobre estos procesos se multiplicó por 7 (de 2 a 14 menciones), concentrándose en rutas latinoamericanas como MAD-MEX (desorden en puerta de embarque, check-in grosero e irrespetuoso), BOG-MAD (boarding caótico sin respetar fila de primera clase, check-in deplorable con upgrades quitados), y LIM-MAD (embarque caótico, equipaje perdido con números de contacto no funcionales). Los pasajeros en vuelos operados bajo acuerdos de codeshare mostraron la mayor reactividad con un spread de 181.4 puntos, sugiriendo problemas de coordinación en ground services con partners, mientras que los viajeros de negocio reaccionaron más positivamente (+19.2 pts) que los de ocio (-4.0 pts), valorando especialmente la mejora en puntualidad crítica para sus necesidades profesionales. Esta presión positiva en Business LH se diluyó al propagarse al radio LH completo (NPS de 9.8 con variación de +4.2 pts, Normal) debido al peso volumétrico de Economy LH (NPS de 4.8 con +2.3 pts, Normal), que mantuvo estabilidad sin cambios significativos, actuando como lastre que absorbió las mejoras de Business y Premium.<br>
<br>
<b>En Premium LH, el NPS llegó a 25.0 (+8.5 pts)</b>, validando la misma mejora operativa generalizada en puntualidad. Los incidentes operativos totales cayeron 47.6% (de 42 a 22), con cancelaciones desapareciendo completamente y retrasos reduciéndose 38.1%. No obstante, esta mejora coexistió con deterioros localizados en producto, especialmente en aircraft interior que impactó negativamente 2.313 ppts según Explanatory Drivers. Las rutas más afectadas por problemas de producto fueron BOG-MAD (NPS de 15.0 con 133 pasajeros, mejora de +2.1 pts a pesar de 4 disrupciones operativas y 6 menciones críticas sobre operación desastrosa y equipaje perdido), EZE-MAD (NPS de 24.4 con 156 pasajeros, el mayor volumen, mejora de +5.8 pts pero con 5 menciones de deterioro en producto como asientos rotos, luces averiadas, baños sin mantenimiento y cabina extremadamente fría), y MAD-NRT (NPS de -30.0 con 20 pasajeros, el peor NPS pero con mejora de +17.4 pts, con 1 mención extremadamente negativa sobre sistema de entretenimiento deficiente, wifi no funcional durante todo el vuelo, azafatas maleducadas con pasajeros japoneses y aseos de más de 20 años de edad). Los pasajeros en vuelos codeshare experimentaron la máxima reactividad con un spread de 320.0 puntos (rango de -120.0 a +200.0 pts), indicando experiencias extremadamente polarizadas que requieren atención especial en gestión de expectativas. Los viajeros de ocio reaccionaron más positivamente (+40.0 pts) que los de negocio (+4.7 pts), siendo más sensibles a la reducción de disrupciones operativas básicas. Por región de residencia, Latinoamérica y Asia mostraron los mayores deterioros (spread de 111.7 pts), coherente con los problemas concentrados en BOG-MAD, EZE-MAD y MAD-NRT. Esta mejora en Premium LH, junto con Business LH, se diluyó al agregarse con Economy LH Normal, resultando en un radio LH estable que, sin embargo, tuvo suficiente peso volumétrico para dominar sobre el deterioro del radio SH.<br>
<br>
<b>En el radio SH, el NPS cayó a 30.2 (-0.9 pts)</b>, resultado de una sinergia negativa donde ambas cabinas (Economy y Business) experimentaron deterioro simultáneo. Este deterioro se explica por un efecto mixto de mejora operativa versus deterioro en servicio. Aunque los incidentes operativos totales mejoraron 36.6% (de 112 a 71), con cancelaciones cayendo 27.6% (de 29 a 21), retrasos disminuyendo 17.9% (de 39 a 32) y desvíos desapareciendo completamente (de 5 a 0), la percepción de puntualidad mejoró solo marginalmente (impacto de +0.768 ppts según Explanatory Drivers). Esta mejora operativa fue neutralizada por un deterioro significativo en drivers de producto y servicio. El deterioro en percepción precio-valor impactó negativamente 2.431 ppts en Economy y 0.899 ppts en Business según Explanatory Drivers, con una paradoja notable: la satisfacción con el precio subió 68.10 puntos en Economy y 62.25 puntos en Business, pero el impacto en NPS fue negativo, indicando que los clientes perciben "pagar lo mismo o más por un servicio degradado". Este deterioro se materializó en tres frentes específicos: la política de equipaje de mano para pasajeros del Grupo 4, con quejas multiplicándose por 2.5 (de 2 a 5+ menciones) en rutas como LCG-MAD ("nos bajaron la maleta a bodega simplemente por pertenecer al grupo 4"), BRU-MAD ("pagué billete caro para tener equipaje accesible") y GVA-MAD ("no había espacio arriba de mi silla, me dijo de poner a mis pies"); el trato del personal, con quejas aumentando 60% (de 8 a 12 menciones) en rutas como EAS-MAD ("tripulación llegó tarde riéndose, nos hicieron esperar 20 min con frío"), LIN-MAD ("agente del mostrador muy desagradable, me quitó maleta de mala manera") y DUS-MAD ("azafata respondió con tono irritado, me despidió con ya basta"); y el aircraft interior, especialmente crítico en Business con impacto de -1.271 ppts según Explanatory Drivers, con quejas aumentando 50% (de 8 a 12 menciones) sobre asientos estrechos en rutas como MAD-VCE, MAD-SVQ y DUS-MAD, con pasajeros de Business reportando "asientos de Economy vendidos como Business". El Load Factor aumentó 3.1 puntos (de 80.27% a 83.37%), generando mayor presión en compartimentos superiores y amplificando los problemas de equipaje de mano. Los pasajeros agrupados por región de residencia mostraron la mayor reactividad con un spread de 166.0 puntos en Economy y 175.0 puntos en Business, con clientes europeos siendo los más sensibles a las políticas de equipaje de mano y al trato del personal. El tipo de flota fue determinante en Business con un spread de 283.3 puntos, validando que el aircraft interior es una causa tangible del deterioro. Este deterioro del radio SH fue superado por el peso volumétrico y la mejora operativa del radio LH, resultando en la anomalía positiva del Global.<br>
<br>
<b>Dentro del radio SH, la compañía YW experimentó caídas severas tanto en Economy (-9.2 pts a NPS de 25.0) como en Business (-10.2 pts a NPS de 16.4)</b>, mientras que IB mantuvo estabilidad en ambas cabinas (Economy: NPS de 32.8 con +3.6 pts Normal; Business: NPS de 36.7 con +0.8 pts Normal). En Economy YW, el deterioro se concentró en puntualidad, con el OTP15 empeorando 5.3 puntos (de 88.77% a 83.43%), generando un impacto de -4.266 ppts según Explanatory Drivers, con la satisfacción cayendo 6.38 puntos. Los problemas de equipaje se intensificaron con el Mishandling empeorando 3.3 puntos (de 18.19 a 21.48), manifestándose en 8 menciones explícitas (33% más que el período anterior) en rutas como BLQ-MAD ("equipaje no embarcado, 4 días después sin saber dónde está", con 2 casos documentados), BCN-SXB ("nos arruinaron las vacaciones, tirados sin ropa de frío", NPS 1) y LIS-MAD ("maletas tardaron 40min en salir, perdí autobús, 8h más en aeropuerto", NPS 0). La arrivals experience deterioró 2.126 ppts según Explanatory Drivers, con problemas específicos en T4S MAD donde el control de pasaportes tomaba 40-60 minutos y las maletas tardaban 40 minutos en salir. Los viajeros de negocio fueron significativamente más reactivos (NPS_diff de -11.3 pts) que los de ocio (+0.6 pts prácticamente neutral), siendo más sensibles a retrasos y conexiones perdidas. Por región de residencia, el spread de 163.5 puntos (rango de -63.5 a +100.0 pts) indicó que residentes de Europa continental fueron los más afectados negativamente. En Business YW, el deterioro fue aún más severo con el OTP15 empeorando 5.34 puntos, generando un impacto de -5.745 ppts según Explanatory Drivers, con la satisfacción con puntualidad cayendo brutalmente 23.28 puntos, la caída más severa de todos los segmentos. El deterioro de servicio de tripulación impactó negativamente 3.425 ppts según Explanatory Drivers, con la satisfacción cayendo 13.0 puntos. La mayor ocupación de vuelos, con el Load Factor aumentando 4.40 puntos (de 74.72% a 79.12%), generó un impacto negativo de -0.562 ppts según Explanatory Drivers, deteriorando la experiencia percibida con menos espacio y menos atención personalizada. Estas presiones negativas de YW en ambas cabinas se transfirieron directamente al agregado SH, a pesar de que IB amortiguó parcialmente el impacto con su estabilidad, resultando en Economy SH cayendo a 30.2 (-0.7 pts) y Business SH cayendo a 30.4 (-3.1 pts), ambas con anomalías negativas que se reforzaron sinérgicamente en el radio SH.<br>
<br>
<b>A nivel Global, el deterioro en percepción precio-valor también se manifestó en rutas LH específicas a través de sobreventa y overbooking</b>, impactando negativamente 1.833 ppts según Explanatory Drivers a pesar de que la satisfacción con el precio subió 29.22 puntos. Este fenómeno se concentró en 10 rutas trianguladas con evidencia de NCS y verbatims: BOS-MAD (NPS de -9.1 con deterioro de -20.2 pts, 22 pasajeros, con quejas de "se vendieron más asientos de los que tocaban, lo que tiene ser la única vía directa Boston-Madrid"), JFK-MAD (NPS de -8.6 con mejora de +3.4 pts, 35 pasajeros, con reportes de "vendieron mi asiento, lo regalaron a otro cliente delante de mis propios ojos"), MAD-SCL (NPS de -17.2 con deterioro de -6.8 pts, 64 pasajeros, con menciones de "vuelo estaba sobrevendido, pasajeros agresivos"), MAD-SJU (NPS de 0.0 con deterioro de -17.5 pts, 24 pasajeros, con problemas de equipaje y servicio), MAD-SDQ (NPS de -12.8 con mejora de +0.6 pts, 39 pasajeros, con servicio y retrasos), MAD-SJO (NPS de 18.2 con mejora de +24.6 pts, 33 pasajeros, con mejora operativa pero problemas residuales de conexiones), MAD-NRT (NPS de -30.0 con mejora de +17.4 pts, 20 pasajeros, con servicio y sensibilidad cultural), MAD-SVQ (NPS de 0.0 con deterioro de -21.9 pts, 28 pasajeros SH, con handling crítico), BOG-MAD (NPS de 15.0 con mejora de +2.1 pts, 133 pasajeros, con servicio crítico) y EZE-MAD (NPS de 24.4 con mejora de +5.8 pts, 156 pasajeros, con servicio durante disrupciones). Los patrones geográficos fueron claros: Norteamérica (JFK, BOS) concentró problemas de sobreventa y overbooking, Caribe (SJU, SDQ, SJO) experimentó problemas de equipaje y servicio, Sudamérica (EZE, BOG, SCL) con alto volumen de 353 pasajeros en 3 rutas combinó servicio y retrasos, y Asia-Pacífico (NRT) mostró problemas de servicio con sensibilidad cultural. Los pasajeros agrupados por acuerdos de codeshare mostraron la mayor reactividad global con un spread de 125.1 puntos (rango de -39.4 a +85.7 pts), seguidos por región de residencia con spread de 120.7 puntos (rango de -32.1 a +88.6 pts), validando que Norteamérica y Asia-Pacífico fueron más sensibles a sobreventa y servicio, mientras que el tipo de flota mostró reactividad media con spread de 54.5 puntos. El propósito del viaje (Business/Leisure) no fue un factor diferenciador significativo con spread de solo 0.1 puntos, indicando que las causas operativas y de producto afectaron transversalmente.<br>
<br>
La convergencia de <b>LH (NPS de 9.8 con variación Normal de +4.2 pts)</b> dominando por volumen y mejora operativa sobre <b>SH (NPS de 30.2 con anomalía negativa de -0.9 pts)</b> produjo el resultado positivo del Global. El mecanismo causal fue claro: LH dominó por volumen y mejora operativa en cabinas premium (Business +15.6 pts, Premium +8.5 pts), con la reducción drástica de cancelaciones (60.1% menos) y retrasos (14.2% menos) generando percepción de mejora en puntualidad (impacto de +0.949 ppts según Explanatory Drivers Global), mientras que el deterioro en producto y servicio (sobreventa impactando -1.833 ppts, ground services en LH Business, políticas de equipaje y trato del personal en SH) limitó la magnitud de la mejora. SH aportó negativamente con su anomalía de -0.9 pts por problemas de producto (equipaje, trato personal, aircraft interior), pero fue superado por el peso de LH, resultando en la anomalía positiva del Global de +0.8 pts.<br>
<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>Economy SH: Transferencia negativa desde YW neutraliza estabilidad de IB</u></b><br>
El NPS de la cabina cayó a 30.2 (-0.7 pts) como resultado directo del deterioro de YW (NPS de 25.0 con caída de -9.2 pts), mientras IB mantuvo estabilidad (NPS de 32.8 con +3.6 pts Normal). El deterioro operativo de YW se concentró en puntualidad, con el OTP15 empeorando 5.3 puntos (de 88.77% a 83.43%) y generando un impacto de -4.266 ppts según Explanatory Drivers, con la satisfacción cayendo 6.38 puntos. Los problemas de equipaje se intensificaron con el Mishandling empeorando 3.3 puntos (de 18.19 a 21.48), manifestándose en 8 menciones explícitas (33% más que el período anterior) especialmente en rutas como BLQ-MAD con 3 menciones críticas (2 casos de equipaje no embarcado con "4 días sin saber dónde está" y 1 caso de retraso de más de 1 hora esperando tripulación de repuesto con NPS 0), LIS-MAD con 2 menciones (maletas tardando 40 minutos en salir, perdiendo autobús y quedando 8 horas más en aeropuerto con NPS 0, más personal maleducado), MAD-SVQ con 2 menciones (pérdida de conexión por retraso de 2 horas sin espera con NPS 0, más comida vegetariana pésima), BCN-SXB con 1 mención crítica de equipaje perdido que arruinó vacaciones (NPS 1), y MAD-TLS con 2 menciones de equipaje perdido más cargos adicionales de 67 euros. La arrivals experience deterioró 2.126 ppts según Explanatory Drivers, con problemas específicos en T4S MAD donde el control de pasaportes tomaba 40-60 minutos y las maletas tardaban 40 minutos en salir. Los viajeros de negocio fueron significativamente más reactivos (NPS_diff de -11.3 pts) que los de ocio (+0.6 pts prácticamente neutral), siendo más sensibles a retrasos y conexiones perdidas. Por región de residencia, el spread de 163.5 puntos (rango de -63.5 a +100.0 pts) indicó que residentes de Europa continental fueron los más afectados negativamente. IB amortiguó parcialmente el impacto con su estabilidad, evitando una caída mayor del agregado, pero no pudo evitar que YW transfiriera su anomalía negativa a la cabina.<br>
<br>
<b><u>Business SH: Transferencia negativa severa desde YW con deterioro múltiple</u></b><br>
El NPS de la cabina cayó a 30.4 (-3.1 pts) como resultado directo del deterioro severo de YW (NPS de 16.4 con caída de -10.2 pts), mientras IB mantuvo estabilidad (NPS de 36.7 con +0.8 pts Normal). El deterioro de YW fue más severo que en Economy debido a la combinación de múltiples factores: puntualidad con el OTP15 empeorando 5.34 puntos y generando un impacto de -5.745 ppts según Explanatory Drivers, con la satisfacción con puntualidad cayendo brutalmente 23.28 puntos (la caída más severa de todos los segmentos); servicio de tripulación impactando negativamente 3.425 ppts según Explanatory Drivers con la satisfacción cayendo 13.0 puntos; y mayor ocupación de vuelos con el Load Factor aumentando 4.40 puntos (de 74.72% a 79.12%), generando un impacto negativo de -0.562 ppts según Explanatory Drivers al deteriorar la experiencia percibida con menos espacio y menos atención personalizada. Aunque la herramienta de rutas falló por error técnico, la evidencia parcial desde incidentes operativos identificó rutas afectadas como LCG-MAD (mencionada en cascada operacional del 24 de diciembre), LCG-SCQ (afectada por transporte terrestre por desvío previo) y conexiones desde y hacia LEU (múltiples desvíos a ILD por meteorología adversa del 19 al 21 de diciembre). Los perfiles reactivos mostraron alta sensibilidad a puntualidad (caída de -23.28 ppts en satisfacción, la más severa), sensibilidad a servicio de tripulación (caída de -13.0 ppts) y sensibilidad a ocupación (impacto negativo de Load Factor de -0.562 ppts a pesar de mayor satisfacción de +8.81 ppts, sugiriendo preferencia por vuelos menos llenos). IB amortiguó el impacto con su estabilidad, pero no pudo evitar que YW transfiriera su anomalía negativa severa a la cabina.<br>
<br>
<b><u>Economy LH: Estabilidad sin cambios significativos</u></b><br>
El NPS se mantuvo en 4.8 (+2.3 pts) dentro del rango Normal, sin cambios significativos detectados. El período actual mantuvo performance estable sin anomalías detectables, sin drivers SHAP anómalos ni análisis detallado disponible. Esta estabilidad actuó como lastre que absorbió las mejoras de Business y Premium LH al agregarse al radio LH completo.<br>
<br>
<b><u>Business LH: Mejora operativa dominante compensa deterioro en ground services</u></b><br>
El NPS alcanzó 34.5 (+15.6 pts) impulsado por la mejora operativa significativa en puntualidad que generó un impacto de 5.650 ppts según Explanatory Drivers. El OTP15 mejoró 4.2 puntos (de 80.04% a 84.27%), mientras que las cancelaciones desaparecieron completamente (de 6 a 0 incidentes) y los retrasos cayeron 38.1% (de 21 a 13 incidentes), totalizando una reducción de 47.6% en incidentes totales (de 42 a 22). Esta mejora eliminó las quejas masivas del período anterior, donde rutas como BOG-MAD reportaban "casi dos horas de retraso con 3 razones diferentes", JFK-MAD sufría "retraso de 3 horas con pésima comunicación" (NPS 0), DOH-MAD experimentaba "retraso impresionante en Doha de 4 horas sin noticias" (NPS 3) y EZE-MAD tenía "el avión salió desde Buenos Aires con demora" (NPS 0). En el período actual, estas quejas prácticamente desaparecieron, con solo 2 menciones menores en MAD-SDQ de "una hora" (NPS 9) y MAD-SCL de "una hora de retraso" (NPS 5). Sin embargo, esta mejora operativa fue parcialmente contrarrestada por un deterioro significativo en procesos de ground services, con boarding impactando negativamente 3.423 ppts y check-in 1.540 ppts según Explanatory Drivers, combinando un impacto total negativo de -4.963 ppts. El número de quejas críticas sobre estos procesos se multiplicó por 7 (de 2 a 14 menciones), concentrándose en rutas latinoamericanas como MAD-MEX con 3 verbatims críticos (desorden en puerta de embarque con maletas, 2 máquinas con una que no sirve y poco personal ayudando; no respeto de reserva de asientos en business; personal de check-in muy grosero e irrespetuoso que se burló con NPS 5), BOG-MAD con 3 verbatims críticos (no respetan la fila de primera clase y el abordaje es un caos con NPS 10; check-in deplorable con upgrade quitado y movidos a turista con NPS 5; desde inicio en counter todo deplorable), LIM-MAD con 2 verbatims críticos (embarque fue caótico con NPS 0; maleta no llegó, números Lima no funcionan, tampoco en ida con NPS 0), GIG-MAD y GRU-MAD con 3 verbatims críticos combinados (equipaje no cargado, 3 días sin maletas con NPS 0; invasión de clase económica a business con NPS 1; no reconocieron nivel tarjeta Infinita con NPS 9), EZE-MAD con 2 verbatims críticos (equipaje 7 días perdido con NPS 0; cambios de asientos unilaterales con NPS 5) y MAD-SJU con 1 verbatim crítico (no llamaron a business primero, abordan demasiado rápido con NPS 8). Los pasajeros en vuelos operados bajo acuerdos de codeshare mostraron la mayor reactividad con un spread de 181.4 puntos (rango de -125.0 a +56.4 pts), sugiriendo problemas de coordinación en ground services con partners. El tipo de flota mostró alta reactividad con spread de 132.6 puntos (rango de -7.6 a +125.0 pts), con evidencia cualitativa de múltiples menciones de "avión antiguo" en MAD-MEX versus "A350 nuevo" en BOG-MAD positivo. Los viajeros de negocio reaccionaron más positivamente (+19.2 pts) que los de ocio (-4.0 pts), valorando especialmente la mejora en puntualidad crítica para sus necesidades profesionales. El balance neto fue que la mejora operativa (impacto de +5.650 ppts) compensó ampliamente el deterioro de ground services (impacto de -4.963 ppts), resultando en la anomalía positiva de +15.6 pts.<br>
<br>
<b><u>Premium LH: Mejora operativa generalizada con deterioros localizados en producto</u></b><br>
El NPS llegó a 25.0 (+8.5 pts) validando la misma mejora operativa generalizada en puntualidad. Los incidentes operativos totales cayeron 47.6% (de 42 a 22), con cancelaciones desapareciendo completamente (de 6 a 0) y retrasos reduciéndose 38.1% (de 21 a 13). El OTP15 mejoró 4.2 puntos (de 80.04% a 84.27%). No obstante, esta mejora coexistió con deterioros localizados en producto, especialmente en aircraft interior que impactó negativamente 2.313 ppts según Explanatory Drivers (con satisfacción cayendo 8.61 puntos). Las rutas más afectadas por problemas de producto fueron BOG-MAD (NPS de 15.0 con 133 pasajeros, el segundo mayor volumen, mejora de +2.1 pts a pesar de 4 disrupciones operativas específicas y 6 menciones críticas sobre operación desastrosa en El Dorado, vuelo retrasado más maletas perdidas, embarque caótico, equipaje extraviado de 8 maletas, retrasos sin explicación y tripulación poco atenta), EZE-MAD (NPS de 24.4 con 156 pasajeros, el mayor volumen, mejora de +5.8 pts pero con 5 menciones de deterioro en producto como estado deficiente del avión con asientos rotos y luces averiadas con NPS 2, baños sin mantenimiento con NPS 6, cabina fría más comida mala con NPS 3), y MAD-NRT (NPS de -30.0 con 20 pasajeros, el peor NPS pero con mejora de +17.4 pts, con 1 mención extremadamente negativa sobre sistema de entretenimiento deficiente sin cámaras y bluetooth no funcional, wifi no funcional durante todo el vuelo, azafatas maleducadas con pasajeros japoneses y aseos del avión parecían de más de 20 años de edad con NPS 0). Los drivers SHAP positivos como check-in (+3.600 ppts), journey preparation support (+3.311 ppts), arrivals experience (+2.002 ppts) y cabin crew (+1.776 ppts) no tuvieron respaldo cualitativo en verbatims, sugiriendo que la mejora se debe más a la ausencia de problemas graves (reducción de disrupciones) que a mejoras activas percibidas por los clientes. Los pasajeros en vuelos codeshare experimentaron la máxima reactividad con un spread de 320.0 puntos (rango de -120.0 a +200.0 pts), indicando experiencias extremadamente polarizadas que requieren atención especial en gestión de expectativas y comunicación. Por región de residencia, el spread de 111.7 puntos (rango de -50.0 a +61.7 pts) mostró que Latinoamérica (coherente con problemas en BOG-MAD y EZE-MAD) y Asia (coherente con incidente grave en MAD-NRT) fueron las más afectadas negativamente. El tipo de flota mostró spread de 91.3 puntos (rango de -73.3 a +18.0 pts), con aeronaves específicas con peor estado de mantenimiento generando mayor insatisfacción, coherente con el impacto negativo de aircraft interior de -2.313 ppts. Los viajeros de ocio reaccionaron más positivamente (+40.0 pts) que los de negocio (+4.7 pts), siendo más sensibles a la reducción de disrupciones operativas básicas como cancelaciones y retrasos prolongados.
```

---

✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** 0
- **Pasos de análisis:** 7
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*



**ANÁLISIS DIARIO SINGLE:**
📅 2025-12-23 to 2025-12-23:
## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 29.4 (+6.4 pts)</b> con respecto a la media de los últimos 7 días, impulsado por una mejora operativa excepcional en el radio Long Haul que compensó ampliamente la estabilidad con tensiones internas del Short Haul.<br>
<br>
<b>Long Haul: Motor de la Mejora Global</b><br>
<br>
En <b>Long Haul</b>, el NPS alcanzó <b>22.3 (+15.3 pts)</b>, el nivel más alto del período analizado, resultado de una mejora operativa generalizada sin precedentes que benefició transversalmente a las tres cabinas. El día previo a Nochebuena presentó condiciones excepcionales: la puntualidad subió 8.3 puntos alcanzando el 91.18%, los incidentes de equipaje se redujeron en 4.41 puntos hasta 17.6 casos, las conexiones perdidas bajaron 0.31 puntos hasta el 0.38%, y la ocupación disminuyó 4.43 puntos hasta el 90.11%, generando mayor espacio y confort para los pasajeros. Esta sinergia operativa se reflejó en las tres cabinas: Economy LH subió 15.6 puntos hasta un NPS de 19.1, Business LH mejoró 8.9 puntos alcanzando 33.3, y Premium LH aumentó 12.8 puntos también hasta 33.3. El feedback cualitativo validó estas mejoras, con el 70% de los comentarios en Premium siendo promotores que elogiaron la puntualidad, el servicio y la comodidad.<br>
<br>
Sin embargo, esta mejora sistémica coexistió con problemas severos localizados en rutas sudamericanas que generaron experiencias extremadamente negativas para un subconjunto de pasajeros. En Economy LH, las rutas <b>GRU-MAD</b> (NPS de –45.5 con 11 pasajeros), <b>MAD-SJO</b> (NPS de –16.7 con 6 pasajeros) y <b>EZE-MAD</b> (NPS de –7.1 con 14 pasajeros) concentraron múltiples quejas: retrasos superiores a 3 horas con pasajeros atrapados en el avión sin explicación ni servicio, robos y daños de equipaje facturado sin seguimiento posterior, asientos incómodos para vuelos de 13 horas, y paradas técnicas no planificadas como el desvío a Santo Domingo en la ruta SJO-MAD. Adicionalmente, se reportó un incidente crítico en el vuelo IB0155 MAD-BOG que fue reprogramado como IB159 con un retraso de 7 horas y 31 minutos, generando 104 conexiones perdidas en Bogotá. Los pasajeros más sensibles fueron los residentes de <b>América Sur</b> (NPS de 0.0 con 30 encuestas en Economy LH) y los que volaron en las flotas <b>A350 next</b> (NPS de 4.2 con 24 encuestas) y <b>A332</b> (NPS de 5.0 con 20 encuestas), ambas operando principalmente rutas largas a Sudamérica. Los vuelos operados bajo código compartido con <b>LATAM</b> mostraron el peor desempeño con un NPS de –50.0 en Economy LH y problemas operativos específicos de coordinación.<br>
<br>
A pesar de estos problemas localizados, el volumen de experiencias positivas en el resto de la red fue abrumador. Rutas como <b>BOG-MAD</b> (NPS de +43.5 con 23 pasajeros en Economy), <b>LIM-MAD</b> (NPS de +28.6 con 7 pasajeros en Economy), <b>MAD-UIO</b>, <b>EZE-MAD</b> y <b>GRU-MAD</b> (todas con NPS de 100.0 en Premium) compensaron ampliamente los incidentes negativos. Esta mejora del Long Haul, que representa el 28% del volumen global con 148 encuestas, fue suficientemente fuerte para arrastrar al resultado global hacia una anomalía positiva de 6.4 puntos, a pesar de que el Short Haul se mantuvo estable.<br>
<br>
<b>Short Haul: Estabilidad Aparente con Volatilidad Interna</b><br>
<br>
En <b>Short Haul</b>, el NPS se mantuvo en <b>32.2 (+1.7 pts)</b> dentro del rango normal, pero este resultado oculta tensiones operativas significativas a nivel de compañía que se cancelaron mutuamente. En Economy SH, <b>IB</b> experimentó una anomalía negativa con un NPS de 32.2 (–0.8 pts) debido a tres problemas convergentes: fallas sistémicas en vuelos operados bajo código compartido con LATAM, AA y BA que registraron un NPS de 0.0 frente al 31.1 de los vuelos propios de IB, un desempeño deficiente de la flota A320 con un NPS de 3.6 (28.6 puntos por debajo del promedio del día) y de la flota A333 con NPS de 0.0, generando 8 menciones de percepción de servicio equivalente a aerolíneas de bajo coste por espacio reducido entre asientos y ausencia de agua gratuita, y una política restrictiva de equipaje de mano con más de 12 menciones de facturación forzada de equipaje de cabina que causó pérdidas de tiempo superiores a 60 minutos. El caso más crítico fue el vuelo IB6659 operado por LATAM en la ruta LIS-MAD vía Lima, donde el sistema de migraciones de Perú no reconoció el número de vuelo de IB, impidiendo el embarque del pasajero con un coste de 1,830.49 euros en nuevo billete. Las rutas más afectadas fueron <b>LIS-MAD</b> (NPS de 0.0 con 9 pasajeros), <b>ARN-MAD</b> (NPS de 0.0 con 5 pasajeros), <b>BIO-MAD</b> (NPS de 0.0 con 10 pasajeros) y <b>DSS-MAD</b> (NPS de 0.0 con 4 pasajeros). Sin embargo, esta anomalía negativa de IB fue completamente diluida por la estabilidad de <b>YW</b>, que mantuvo un NPS de 31.1 (+5.6 pts) sin incidentes significativos, resultando en una cabina Economy SH normal.<br>
<br>
En Business SH, se produjo una cancelación de efectos opuestos aún más marcada. <b>IB</b> sufrió una anomalía negativa con un NPS de 37.5 (–1.4 pts) causada por una huelga en el aeropuerto de Madrid el 23 de diciembre. Se registraron 9 incidentes operativos que incluyeron 7 retrasos, 2 cancelaciones y otras incidencias, con el 71% de las rutas afectadas teniendo Madrid como destino. Dos pasajeros mencionaron explícitamente la huelga en sus comentarios: uno en la ruta LHR-MAD reportó que el equipaje de su acompañante se retrasó debido a la huelga, y otro en DSS-MAD indicó que esperó 3 horas sin que se comunicara que el problema era una huelga de Iberia. Las rutas <b>BRU-MAD</b> (NPS de 0.0), <b>BCN-MAD</b> (NPS de 40.0), <b>DUS-MAD</b> (NPS de 50.0) y <b>LHR-MAD</b> (NPS de 60.0) mostraron el mayor impacto. Los pasajeros europeos no españoles fueron los más afectados con un NPS de 16.7, mientras que los españoles mostraron mayor tolerancia con un NPS de 62.5. La flota A321 concentró el mayor impacto negativo con un NPS de 14.3, representando el 29% de las encuestas de Business IB. Simultáneamente, <b>YW</b> experimentó una anomalía positiva con un NPS de 31.2 (+15.2 pts) impulsada por mejoras operativas generalizadas: la puntualidad subió 2.98 puntos, los incidentes de equipaje bajaron 4.54 puntos, las conexiones perdidas se redujeron 0.23 puntos, y la ocupación disminuyó 5.11 puntos. El 70% de los comentarios de YW fueron promotores que elogiaron la atención del personal y la organización. Estos efectos opuestos se anularon completamente, resultando en una cabina Business SH normal con un NPS de 35.0 (+3.1 pts) que oculta una volatilidad interna de 16.6 puntos entre las dos compañías.<br>
<br>
La convergencia de <b>Long Haul (+15.3 pts)</b> en anomalía positiva y <b>Short Haul (+1.7 pts)</b> en estado normal, con el primero representando el 28% del volumen y el segundo el 72%, produjo el resultado global de 29.4 puntos con una mejora de 6.4 puntos. El Long Haul dominó el resultado agregado debido a la magnitud excepcional de su mejora operativa, que fue suficientemente fuerte para arrastrar al global hacia una anomalía positiva a pesar de que el Short Haul se mantuvo estable con tensiones internas compensadas.<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>ECONOMY SH: Problemas de IB Diluidos por Estabilidad de YW</u></b><br>
La cabina Economy SH mantuvo un <b>NPS de 31.9 (+1.4 pts)</b> dentro del rango normal, pero este resultado enmascara una anomalía negativa de <b>IB</b> con un NPS de 32.2 (–0.8 pts) que fue completamente diluida por la estabilidad de <b>YW</b> con un NPS de 31.1 (+5.6 pts). IB enfrentó tres problemas convergentes: fallas sistémicas en vuelos de código compartido donde LATAM, AA y BA registraron un NPS de 0.0 frente al 31.1 de los vuelos propios de IB, con un caso crítico en el vuelo IB6659 operado por LATAM en la ruta LIS-MAD vía Lima donde el sistema de migraciones peruano no reconoció el número de vuelo impidiendo el embarque con un coste de 1,830.49 euros; un desempeño deficiente de las flotas A320 con NPS de 3.6 (24% del volumen, 28.6 puntos por debajo del promedio) y A333 con NPS de 0.0, generando 8 menciones de percepción de servicio equivalente a bajo coste por espacio reducido y ausencia de agua gratuita en rutas como LIS-MAD, MAD-VIE, LIN-MAD, FCO-MAD y DSS-MAD; y una política restrictiva de equipaje de mano con más de 12 menciones de facturación forzada de equipaje de cabina que causó pérdidas de tiempo superiores a 60 minutos en rutas como MAD-VIE, MAD-ORY, LIS-MAD, BRU-MAD, MAD-VCE y AMS-MAD, asociada a una ocupación del 87.11%. Las rutas más críticas fueron <b>LIS-MAD</b> (NPS de 0.0 con 9 pasajeros y 5 menciones en comentarios), <b>ARN-MAD</b> (NPS de 0.0 con 5 pasajeros), <b>BIO-MAD</b> (NPS de 0.0 con 10 pasajeros), <b>DSS-MAD</b> (NPS de 0.0 con 4 pasajeros) y <b>MAD-PRG</b> (NPS de 0.0 con 6 pasajeros). Los perfiles más reactivos fueron los pasajeros que volaron en flotas A333 y A320, los que utilizaron vuelos de código compartido con LATAM, AA y BA, y los residentes de Asia con un NPS de –66.7, Europa con 30.4 y España con 40.2. IB representa el 94% del volumen de Economy SH con 236 encuestas, pero su deterioro fue completamente compensado por el desempeño estable de YW.<br>
<br>
<b><u>BUSINESS SH: Huelga en Madrid de IB Cancelada por Mejora Operativa de YW</u></b><br>
La cabina Business SH registró un <b>NPS de 35.0 (+3.1 pts)</b> dentro del rango normal, resultado de una cancelación de efectos opuestos entre <b>IB</b> con un NPS de 37.5 (–1.4 pts) e <b>YW</b> con un NPS de 31.2 (+15.2 pts). IB sufrió el impacto de una huelga en el aeropuerto de Madrid el 23 de diciembre que generó 9 incidentes operativos incluyendo 7 retrasos, 2 cancelaciones y otras incidencias, con el 71% de las rutas afectadas teniendo Madrid como destino. La puntualidad bajó 0.43 puntos con respecto al baseline. Dos pasajeros mencionaron explícitamente la huelga: uno en LHR-MAD reportó que el equipaje de su acompañante se retrasó debido a la huelga y tuvo que ser entregado en su casa, y otro en DSS-MAD indicó que esperó 3 horas sin que se comunicara que el problema era una huelga de Iberia, destacando la falta de comunicación proactiva sobre la situación. Las rutas más afectadas fueron <b>BRU-MAD</b> con NPS de 0.0 por servicio en tierra deficiente, <b>BCN-MAD</b> con NPS de 40.0 por impuntualidad, <b>DUS-MAD</b> con NPS de 50.0 afectado por la huelga, y <b>LHR-MAD</b> con NPS de 60.0 por equipaje retrasado. Los pasajeros europeos no españoles fueron los más sensibles con un NPS de 16.7, mientras que los españoles mostraron mayor tolerancia con un NPS de 62.5, posiblemente por mayor familiaridad con disrupciones locales. La flota A321 concentró el mayor impacto negativo con un NPS de 14.3, representando el 29% de las encuestas de Business IB. Simultáneamente, YW experimentó una mejora operativa excepcional con la puntualidad subiendo 2.98 puntos hasta el 87.98%, los incidentes de equipaje bajando 4.54 puntos hasta 15.91 casos, las conexiones perdidas reduciéndose 0.23 puntos hasta el 0.17%, y la ocupación disminuyendo 5.11 puntos hasta el 53.06%, generando mayor espacio y confort. El 70% de los comentarios de YW fueron promotores que elogiaron la atención del personal, la organización y la confiabilidad de la compañía, con menciones positivas en rutas como MAD-XRY, MAD-TRN, PMI-VLL, IBZ-VLC y BOD-MAD. Estos efectos opuestos se anularon completamente, resultando en una cabina Business SH estable que oculta una volatilidad interna de 16.6 puntos entre las dos compañías.<br>
<br>
<b><u>ECONOMY LH: Mejora Operativa Excepcional con Problemas Localizados en Sudamérica</u></b><br>
La cabina Economy LH alcanzó un <b>NPS de 19.1 (+15.6 pts)</b> en anomalía positiva, impulsada por una mejora operativa generalizada excepcional que benefició transversalmente a todas las cabinas del Long Haul. El día previo a Nochebuena presentó condiciones sin precedentes: la puntualidad subió 8.3 puntos alcanzando el 91.18%, los incidentes de equipaje se redujeron en 4.41 puntos hasta 17.6 casos, las conexiones perdidas bajaron 0.31 puntos hasta el 0.38%, y la ocupación disminuyó 3.68 puntos hasta el 91.1%, generando mayor espacio disponible. Sin embargo, esta mejora global coexistió con problemas severos localizados en rutas sudamericanas. <b>GRU-MAD</b> registró un NPS de –45.5 con 11 pasajeros, concentrando 4 quejas graves sobre retrasos superiores a 3 horas con pasajeros atrapados en el avión sin explicación ni servicio, robos de objetos en equipaje facturado con maletas forzadas por personal del aeropuerto, y daños como ruedas rotas sin seguimiento posterior a pesar de completar los formularios necesarios. <b>MAD-SJO</b> obtuvo un NPS de –16.7 con 6 pasajeros debido a una parada técnica no planificada en Santo Domingo, ausencia de WiFi en avión moderno, y deterioro en la calidad de la comida comparado con viajes anteriores. <b>EZE-MAD</b> alcanzó un NPS de –7.1 con 14 pasajeros por asientos incómodos para vuelos de 13 horas, trato deficiente del personal de cabina con menciones de maltrato serial y trato horrible, y configuración de asientos que no cumplió las expectativas de los pasajeros. <b>MAD-MVD</b> registró un NPS de 0.0 con 4 pasajeros con quejas sobre 13 horas en asientos estrechos donde hasta personas delgadas van incómodas con piernas entumecidas. Los pasajeros más sensibles fueron los residentes de <b>América Sur</b> con un NPS de 0.0 con 30 encuestas, los que volaron en las flotas <b>A350 next</b> con NPS de 4.2 con 24 encuestas y <b>A332</b> con NPS de 5.0 con 20 encuestas, ambas operando principalmente rutas largas a Sudamérica, y los que utilizaron vuelos operados bajo código compartido con <b>LATAM</b> con un NPS de –50.0 con 4 encuestas. A pesar de estos problemas localizados, rutas como <b>BOG-MAD</b> con NPS de +43.5 con 23 pasajeros y <b>LIM-MAD</b> con NPS de +28.6 con 7 pasajeros compensaron ampliamente los incidentes negativos, resultando en la anomalía positiva de 15.6 puntos.<br>
<br>
<b><u>BUSINESS LH: Mejora Operativa Moderada Atenuada por Incidentes Puntuales</u></b><br>
La cabina Business LH registró un <b>NPS de 33.3 (+8.9 pts)</b> en anomalía positiva, impulsada por la misma mejora operativa generalizada que benefició a todas las cabinas del Long Haul: la puntualidad subió 8.3 puntos alcanzando el 91.18%, los incidentes de equipaje se redujeron en 4.37 puntos hasta 17.6 casos, las conexiones perdidas bajaron 0.31 puntos hasta el 0.38%, y la ocupación disminuyó 7.07 puntos hasta el 86.4%, generando menor saturación y mejor confort que Economy. Sin embargo, la magnitud de la mejora fue moderada en comparación con Economy LH debido a factores atenuantes. Se reportó un incidente crítico en el vuelo IB0155 MAD-BOG que fue reprogramado como IB159 con un retraso de 7 horas y 31 minutos, generando 104 conexiones perdidas en Bogotá. Adicionalmente, se identificaron problemas de servicio a bordo y producto en rutas específicas: <b>LIM-MAD</b> alcanzó un NPS de –100.0 con 1 pasajero que reportó que el servicio en clase ejecutiva no era el estándar y la tripulación no se comportó de forma amable cada vez que solicitó algo con cortesía, además de un embarque caótico y equipaje que no llegó sin novedades posteriores; <b>MAD-MVD</b> registró un NPS de 0.0 con 2 pasajeros que indicaron que el avión es viejo y el sistema de video no funciona bien; y <b>BOG-MAD</b> mostró un NPS de 0.0 con 5 pasajeros con problemas operativos, aunque existe una contradicción con 1 pasajero que reportó NPS de 100.0, sugiriendo un desfase temporal en la captura de encuestas. Los pasajeros más sensibles fueron los residentes de <b>Europa</b> con un NPS de 0.0 con 2 encuestas, <b>América Norte</b> con NPS de –100.0 con 1 encuesta, y los que volaron en la flota <b>A350 next</b> con NPS de 0.0 con 3 encuestas. Los vuelos operados bajo código compartido con <b>AA</b> y <b>BA</b> mostraron un NPS de 0.0 con 2 encuestas cada uno, frente al 42.9 de los vuelos propios de IB. A pesar de estos problemas localizados, rutas como <b>BOG-MAD</b>, <b>MAD-ORD</b>, <b>JFK-MAD</b> y <b>MAD-SCL</b> alcanzaron un NPS de 100.0, compensando parcialmente los incidentes negativos y resultando en la anomalía positiva de 8.9 puntos.<br>
<br>
<b><u>PREMIUM LH: Mejora Operativa Robusta con Feedback Predominantemente Positivo</u></b><br>
La cabina Premium LH alcanzó un <b>NPS de 33.3 (+12.8 pts)</b> en anomalía positiva, impulsada por la misma mejora operativa generalizada que benefició a todas las cabinas del Long Haul: la puntualidad subió 8.3 puntos alcanzando el 91.18%, los incidentes de equipaje se redujeron en 4.41 puntos hasta 17.60 casos, las conexiones perdidas bajaron 0.31 puntos hasta el 0.38%, y la ocupación disminuyó 4.43 puntos hasta el 90.11%. El feedback fue predominantemente positivo con el 70% de los comentarios siendo promotores con NPS de 9 a 10 que elogiaron la puntualidad, el servicio, la comodidad y la atención. Sin embargo, es importante notar que el volumen de muestra fue extremadamente bajo con solo 15 encuestas procesadas, lo que limita la confiabilidad estadística del análisis. A pesar de la mejora generalizada, se identificaron incidentes localizados: <b>LIM-MAD</b> registró un NPS de –100.0 con 1 pasajero que reportó un error en la asignación de asiento y un tripulante poco agradable; <b>BOG-MAD</b> alcanzó un NPS de 0.0 con 5 pasajeros con problemas de retrasos, conexiones perdidas y equipaje, incluyendo el incidente crítico del vuelo IB0155 reprogramado como IB159 con 7 horas y 31 minutos de retraso que generó 104 conexiones perdidas; y <b>MAD-SJO</b> obtuvo un NPS de 50.0 con 2 pasajeros que reportaron una espera de 1 hora y 30 minutos por el equipaje. Sin embargo, el excelente desempeño de otras rutas compensó ampliamente estos problemas: <b>MAD-UIO</b> alcanzó un NPS de 100.0 con 1 pasajero que destacó los upgrades y la experiencia premium, <b>EZE-MAD</b> registró un NPS de 100.0 con 1 pasajero que elogió la puntualidad y el servicio, y <b>GRU-MAD</b> obtuvo un NPS de 100.0 con 3 pasajeros que valoraron la comodidad y la atención. Los perfiles más reactivos fueron los viajeros por ocio con un NPS de 42.9 con 14 encuestas representando el 93% de la muestra, mientras que los viajeros de negocios mostraron un NPS de –100.0 con 1 encuesta, indicando que los viajeros corporativos son más vulnerables a las disrupciones operativas. Los residentes de Europa alcanzaron un NPS de 50.0 con 2 encuestas, España 33.3 con 6 encuestas, América Sur 20.0 con 5 encuestas, y América Centro 0.0 con 2 encuestas.
```
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-22 to 2025-12-22:
## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 25.6 (+2.5 pts)</b> con respecto a la media de los últimos 7 días, resultado de una mejora significativa en Long Haul que absorbió completamente el deterioro marginal de Short Haul.<br>
<br>
En <b>Long Haul Economy</b>, el NPS alcanzó <b>13.6 (+10.1 pts)</b>, impulsado por mejoras operativas sustanciales que transformaron la experiencia del cliente. La puntualidad mejoró 6.57 puntos porcentuales hasta el 85.51 por ciento, mientras que los incidentes de equipaje disminuyeron 1.32 puntos y la ocupación se redujo 1.14 puntos porcentuales, generando mayor confort a bordo. Este desempeño operativo se vio reforzado por el excelente rendimiento de las flotas modernas, particularmente el A350 next con NPS de 31.8 y el A333 con NPS de 40.0, que representaron el 35.1 por ciento del volumen y compensaron los problemas de flotas legacy como el A350 C con NPS de menos 10.0 y el A321XLR con NPS de 0.0. Adicionalmente, los vuelos operados directamente por IB alcanzaron un NPS de 18.9, superando en 49.7 puntos al promedio de codeshares con otros partners, donde LATAM registró menos 50.0, Qatar Airways menos 33.3 y British Airways menos 20.0. Las rutas latinoamericanas destacaron positivamente, con GIG-MAD alcanzando NPS de 100.0, GRU-MAD con 40.0 y BOG-MAD con 37.5, mientras que las rutas problemáticas se concentraron en BOS-MAD con menos 16.7, LIM-MAD con menos 10.0 por pérdida de equipaje, MAD-SCL con menos 8.3 por retrasos y conexiones perdidas, y MAD-MVD con 0.0 por cambios de vuelo forzados. Los pasajeros más sensibles fueron los residentes de América Centro con NPS de 33.3 y América Sur con NPS de 31.0, contrastando con los europeos que registraron menos 15.0, evidenciando una dispersión de 133.3 puntos. Esta mejora en Economy LH, que representa el 81.5 por ciento del volumen del radio, dominó el agregado Long Haul a pesar de la caída de Business LH de menos 8.4 puntos, arrastrando finalmente al Global hacia territorio positivo.<br>
<br>
En <b>Long Haul Premium</b>, el NPS subió a <b>30.0 (+9.5 pts)</b>, reforzando la tendencia positiva del radio. Las mismas mejoras operativas que beneficiaron a Economy también impactaron favorablemente a Premium, con puntualidad aumentando 6.57 puntos porcentuales y gestión de equipaje mejorando 1.32 puntos. Los vuelos operados por IB alcanzaron NPS de 62.5, representando el 80 por ciento del volumen, mientras que los codeshares mostraron problemas críticos, especialmente American Airlines con NPS de menos 100.0 en la ruta MAD-ORD, donde se registraron retrasos acumulados de dos días, cobros indebidos de equipaje y problemas de integración entre sistemas IB-AA. La flota A350 destacó con NPS de 75.0, mientras que el A333 registró menos 100.0, evidenciando una dispersión de 175 puntos. Las rutas BOG-MAD y EZE-MAD alcanzaron NPS de 100.0, contrastando con los problemas en MAD-ORD y MAD-MEX, donde el espacio limitado en Premium Economy generó quejas específicas. Los pasajeros de América Centro registraron NPS de 100.0, mientras que aquellos sin región identificada alcanzaron menos 100.0, mostrando una dispersión extrema de 200 puntos. Esta mejora en Premium, aunque representó solo el 5.3 por ciento del volumen Long Haul, complementó el impulso positivo de Economy y ayudó a contrarrestar el deterioro de Business en el agregado del radio.<br>
<br>
En <b>Short Haul Economy IB</b>, el NPS cayó a <b>31.7 (menos 1.3 pts)</b>, arrastrando a la cabina completa hacia territorio negativo. El deterioro se originó en tres problemas convergentes. Primero, la puntualidad empeoró 0.81 puntos porcentuales hasta el 86.55 por ciento y las conexiones perdidas aumentaron 0.19 puntos porcentuales hasta el 1.07 por ciento, respaldado por seis retrasos reportados en incidentes operativos y múltiples quejas de retrasos de una a dos horas en verbatims. Segundo, aunque la métrica global de equipaje mejoró 1.32 puntos, se registraron más de siete incidentes graves en comentarios de clientes, concentrados en hubs específicos como Madrid y Bolonia, incluyendo pérdida de equipaje durante cuatro días sin localizar en BLQ-MAD, esperas de una hora en cintas transportadoras en BRU-MAD, y maletas abolladas tras facturación forzosa en FCO-MAD. Tercero, la política de facturación forzosa de equipaje de mano en vuelos medio vacíos generó frustración significativa, con casos documentados en LCG-MAD, FCO-MAD y CDG-MAD, este último resultando en pérdida de tren y costes adicionales de 70 euros. Las rutas más afectadas fueron GRX-MLN con NPS de menos 33.3 por servicio de tripulación deficiente, ALC-MAD con menos 23.1 por equipaje extraviado, MAD-SVQ con 0.0 por pérdida de conexión tras retraso de dos horas, y BCN-MAD con 21.2 por combinación de retrasos, equipaje y conexiones perdidas. Los viajeros de ocio fueron los más afectados con NPS de 28.7, representando el 93.6 por ciento del volumen y mostrando una brecha de 21.3 puntos respecto a viajeros de negocios. Las flotas CRJ con NPS de 24.1 y A321 con NPS de 14.8 concentraron los problemas, operando las rutas críticas identificadas y evidenciando una dispersión de 50 puntos. Los pasajeros europeos registraron NPS de 23.6, mostrándose más críticos con los problemas de conexiones que los españoles con 36.9, en una dispersión de 89.4 puntos. Esta presión en IB, que representa el 92 por ciento del volumen de Economy SH, se transfirió al agregado de la cabina con NPS de 30.1 (menos 0.5 pts), a pesar de que YW mantuvo estabilidad con NPS de 27.2 (+1.7 pts) actuando como amortiguador parcial. Posteriormente, Economy SH arrastró al radio completo Short Haul hacia NPS de 30.4 (menos 0.1 pts), aunque Business SH con NPS de 33.3 (+1.4 pts) también mitigó parcialmente el impacto.<br>
<br>
En <b>Short Haul Business IB</b>, el NPS registró <b>38.7 (menos 0.2 pts)</b>, evidenciando problemas estructurales que fueron completamente diluidos por YW en el agregado de la cabina. El deterioro se concentró en tres áreas críticas. Primero, la calidad del producto Business en la flota A320 legacy alcanzó NPS de menos 25.0, con una brecha de 85 puntos respecto al A320neo con NPS de 60.0, generando cuatro menciones específicas en verbatims sobre asientos considerados normales donde solo se quita el asiento del medio, comida no acorde a clase ejecutiva, tapas ridículas y espacio insuficiente para estirar piernas por paredes delanteras. Segundo, se registraron once incidentes operativos que incluyeron seis retrasos y dos cancelaciones, destacando un fallo en el sistema de equipajes en la escala de Milán Malpensa, con tres menciones en comentarios de clientes sobre equipaje olvidado durante 72 horas en LHR-MAD, espera de 39 minutos en cinta transportadora también en LHR-MAD, y rotura de andadera en BCN-MAD. Tercero, los pasajeros de codeshare British Airways registraron NPS de 0.0, evidenciando un gap de expectativas al encontrar diferencias significativas con los estándares esperados del partner. Las rutas más afectadas fueron BCN-MAD con NPS de menos 33.3 por equipaje dañado, servicio deficiente y comida inadecuada, CDG-MAD con 0.0 donde los asientos fueron descritos como normales, LHR-MAD con 20.0 por equipaje lento y comida no acorde, y MAD-OSL con 33.3 por espacio limitado en primera fila. Los viajeros de negocios fueron los más críticos con NPS de 0.0, mostrando una brecha de 42.9 puntos respecto a viajeros de ocio con 42.9, mientras que los pasajeros europeos alcanzaron NPS de 41.7, 36.1 puntos por debajo de los españoles con 77.8, en una dispersión extrema de 177.8 puntos. Esta anomalía negativa de IB fue completamente diluida por el desempeño estable de YW con NPS de 21.4 (+5.4 pts), resultando en un agregado de Business SH de 33.3 (+1.4 pts) con estado normal que oculta la volatilidad interna real entre las dos compañías.<br>
<br>
La convergencia de <b>Long Haul (+7.8 pts)</b> y <b>Short Haul (menos 0.1 pts)</b> produjo el resultado positivo del Global. A pesar de que Short Haul representa el 69 por ciento del volumen con 421 encuestas, su deterioro marginal fue 78 veces inferior en magnitud a la mejora de Long Haul, que con el 31 por ciento del volumen y 189 encuestas impuso su signo positivo al agregado. La contribución neta de Long Haul fue de más 2.42 puntos mientras que Short Haul aportó menos 0.07 puntos, consolidando el NPS global en 25.6 con una mejora de 2.5 puntos respecto a la media de los últimos siete días.<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>Economy SH: Deterioro operativo concentrado en IB arrastró la cabina</u></b><br>
La cabina registró <b>NPS de 30.1 (menos 0.5 pts)</b> como resultado directo de la transferencia de problemas desde IB, que con el 92 por ciento del volumen alcanzó NPS de 31.7 (menos 1.3 pts), mientras YW mantuvo estabilidad con NPS de 27.2 (+1.7 pts) actuando como amortiguador parcial. El deterioro de IB se originó en la combinación de puntualidad empeorada 0.81 puntos porcentuales, conexiones perdidas aumentadas 0.19 puntos porcentuales con seis retrasos documentados, más de siete incidentes graves de equipaje concentrados en hubs específicos de Madrid y Bolonia a pesar de la mejora en la métrica global, y una política de facturación forzosa de equipaje de mano que generó pérdida de conexiones y costes adicionales de hasta 70 euros. Las rutas más afectadas fueron GRX-MLN con NPS de menos 33.3, ALC-MAD con menos 23.1, MAD-SVQ con 0.0 y BCN-MAD con 21.2, mientras que los viajeros de ocio con NPS de 28.7 representaron el 93.6 por ciento del volumen mostrando mayor sensibilidad que los de negocios con brecha de 21.3 puntos. Las flotas CRJ con NPS de 24.1 y A321 con NPS de 14.8 concentraron los problemas en una dispersión de 50 puntos, y los pasajeros europeos con NPS de 23.6 fueron más críticos que los españoles con 36.9 en una dispersión de 89.4 puntos.<br>
<br>
<b><u>Business SH: Problemas de producto IB diluidos por YW generan estabilidad aparente</u></b><br>
La cabina alcanzó <b>NPS de 33.3 (+1.4 pts)</b> con estado normal, pero este resultado oculta la volatilidad interna entre IB con NPS de 38.7 (menos 0.2 pts) y YW con NPS de 21.4 (+5.4 pts), donde el desempeño estable de YW diluyó completamente la anomalía negativa de IB. Los problemas de IB se concentraron en la calidad del producto Business de la flota A320 legacy con NPS de menos 25.0, mostrando una brecha de 85 puntos respecto al A320neo con NPS de 60.0, con quejas específicas sobre asientos considerados normales, comida no acorde a clase ejecutiva y espacio insuficiente para piernas. Adicionalmente, once incidentes operativos incluyeron seis retrasos, dos cancelaciones y un fallo en el sistema de equipajes en Milán Malpensa, generando tres menciones de equipaje olvidado durante 72 horas, esperas de 39 minutos en cinta y rotura de andaderas. El gap de expectativas en codeshare British Airways alcanzó NPS de 0.0, evidenciando diferencias con los estándares esperados. Las rutas más afectadas fueron BCN-MAD con NPS de menos 33.3, CDG-MAD con 0.0, LHR-MAD con 20.0 y MAD-OSL con 33.3, mientras que los viajeros de negocios con NPS de 0.0 mostraron una brecha de 42.9 puntos respecto a ocio, y los pasajeros europeos con NPS de 41.7 quedaron 36.1 puntos por debajo de los españoles con 77.8 en una dispersión extrema de 177.8 puntos.<br>
<br>
<b><u>Economy LH: Mejoras operativas y flotas modernas impulsaron el desempeño</u></b><br>
La cabina alcanzó <b>NPS de 13.6 (+10.1 pts)</b> impulsada por mejoras operativas significativas donde la puntualidad aumentó 6.57 puntos porcentuales hasta el 85.51 por ciento, los incidentes de equipaje disminuyeron 1.32 puntos y la ocupación se redujo 1.14 puntos porcentuales generando mayor confort. El excelente desempeño de flotas modernas como el A350 next con NPS de 31.8 y el A333 con NPS de 40.0, que representaron el 35.1 por ciento del volumen, compensó los problemas de flotas legacy como el A350 C con NPS de menos 10.0 y el A321XLR con NPS de 0.0, evidenciando una dispersión de 50 puntos. Los vuelos operados directamente por IB alcanzaron NPS de 18.9 superando en 49.7 puntos al promedio de codeshares con menos 30.8, donde LATAM registró menos 50.0, Qatar Airways menos 33.3 y British Airways menos 20.0 en una dispersión de 150 puntos. Las rutas latinoamericanas destacaron positivamente con GIG-MAD alcanzando NPS de 100.0, GRU-MAD con 40.0 y BOG-MAD con 37.5, mientras que las problemáticas se concentraron en BOS-MAD con menos 16.7, LIM-MAD con menos 10.0 por pérdida de equipaje, MAD-SCL con menos 8.3 por retrasos y MAD-MVD con 0.0 por cambios de vuelo forzados. Los pasajeros de América Centro con NPS de 33.3 y América Sur con NPS de 31.0 contrastaron con los europeos con menos 15.0 en una dispersión de 133.3 puntos, mientras que los viajeros de ocio con NPS de 14.0 representaron el 97.4 por ciento del volumen con dispersión mínima de 6.9 puntos respecto a negocios.<br>
<br>
<b><u>Business LH: Caída sin causa operativa clara requiere investigación adicional</u></b><br>
La cabina registró <b>NPS de 16.0 (menos 8.4 pts)</b> en una paradoja operativa donde las métricas mejoraron significativamente con puntualidad aumentando 6.57 puntos porcentuales, gestión de equipaje mejorando 1.32 puntos y ocupación reducida 2.68 puntos porcentuales, sin correlación aparente con la caída del NPS. La única métrica que empeoró fue conexiones perdidas con aumento de 0.19 puntos porcentuales, pero esta desviación es insuficiente para explicar una caída de 8.4 puntos. La investigación se limitó a métricas operativas sin ejecutar análisis de incidentes operativos, comentarios de clientes, rutas ni perfiles, por lo que se requiere investigación adicional para identificar causas específicas no capturadas por métricas agregadas, posiblemente relacionadas con factores cualitativos como servicio a bordo, catering, atención de tripulación o incidentes puntuales de alta gravedad que no se reflejan en los agregados.<br>
<br>
<b><u>Premium LH: Mejoras operativas con incidentes puntuales en codeshares</u></b><br>
La cabina alcanzó <b>NPS de 30.0 (+9.5 pts)</b> impulsada por las mismas mejoras operativas globales que beneficiaron a Economy, con puntualidad aumentando 6.57 puntos porcentuales, gestión de equipaje mejorando 1.32 puntos y ocupación reducida 1.43 puntos porcentuales. Los vuelos operados por IB alcanzaron NPS de 62.5 representando el 80 por ciento del volumen, mientras que los codeshares mostraron problemas críticos con American Airlines registrando NPS de menos 100.0 en una dispersión de 162.5 puntos. La flota A350 destacó con NPS de 75.0 mientras que el A333 registró menos 100.0 en una dispersión de 175 puntos. Las rutas BOG-MAD y EZE-MAD alcanzaron NPS de 100.0 con menciones de puntualidad y atención destacada, contrastando con MAD-ORD que registró menos 100.0 por retrasos acumulados de dos días, cobros indebidos de equipaje y problemas de integración entre sistemas IB-AA, y MAD-MEX con 50.0 por espacio limitado en Premium Economy. Los pasajeros de América Centro registraron NPS de 100.0 mientras que aquellos sin región identificada alcanzaron menos 100.0 en una dispersión extrema de 200 puntos, evidenciando que los incidentes puntuales en codeshares generaron detractores extremos que atenuaron la mejora potencial de la cabina.
```
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-21 to 2025-12-21:
## 📋 SÍNTESIS EJECUTIVA FINAL

```html
<b>SÍNTESIS EJECUTIVA</b><br>
<br>
La red global registró un <b>NPS de 25.8 (+2.8 pts)</b> con respecto a la media de los últimos 7 días. Sin embargo, esta mejora aparente oculta una <b>paradoja operativa crítica</b>: todas las métricas operativas clave muestran deterioro significativo (Mishandling +5.16 pts, OTP -2.57 pts, 130 conexiones perdidas), lo que sugiere que la anomalía positiva resulta de un baseline anormalmente bajo en la semana previa o de sesgos de composición en las respuestas, más que de una mejora real del servicio.<br>
<br>
En <b>Business SH</b>, el NPS cayó a <b>17.8 (–14.2 pts)</b>, resultado directo del colapso operativo de <b>IB</b>, que registró <b>16.1 (–22.8 pts)</b>. La causa raíz fue la <b>meteorología adversa en el aeropuerto de Lleida-Alguaire (LEU)</b>, que generó 12 cancelaciones masivas (92% de los incidentes del día), desencadenando un efecto cascada en la red doméstica española. Este evento deterioró la puntualidad (OTP15 cayó 4.48 pts hasta 87.13%) y disparó los problemas de equipaje (Mishandling aumentó 4.54 pts hasta 28.29), casi duplicando la tasa normal. Las rutas más afectadas fueron <b>GVA-MAD</b> (NPS 0.0, 3 pasajeros reportando retrasos de 45-50 minutos en entrega de equipaje sin respetar prioridades Business), <b>EAS-MAD</b> (NPS 0.0, con quejas críticas sobre tripulación tardía y pasajeros esperando 20 minutos al frío), <b>CMN-MAD</b> (NPS 0.0, equipaje no entregado incluyendo medicación y documentos valiosos), <b>BLQ-MAD</b> (NPS 0.0, cancelación sin compensación con esperas superiores a 5 horas) y <b>MAD-VGO</b> (NPS 0.0). Los viajeros de negocios fueron los más sensibles (NPS –50.0 vs Leisure 21.4), especialmente aquellos de origen europeo no español (NPS 5.6) y sudamericano (NPS –25.0). La flota A321 mostró el peor rendimiento (NPS –11.1) en operaciones de corto radio. Aunque <b>YW</b> mantuvo estabilidad con <b>21.4 (+5.4 pts, Normal)</b>, su volumen reducido (4 encuestas, 9% del total) fue insuficiente para contrarrestar el peso de IB (41 encuestas, 91% del total), que dominó e impuso su signo negativo al agregado de la cabina. Esta presión en Business SH se neutralizó posteriormente con Economy SH (que mejoró +7.5 pts), resultando en un agregado SH falsamente Normal (+5.3 pts) que oculta la volatilidad extrema entre productos.<br>
<br>
En <b>Premium LH</b>, el NPS cayó a <b>5.9 (–14.7 pts)</b> debido a dos causas raíz convergentes. La primera fue el <b>incidente técnico en el vuelo IB0155 (MAD-BOG del 21 de diciembre)</b>, que resultó en una reprogramación de 24 horas (vuelo trasladado al IB157 del 22 de diciembre), afectando al 23.5% de los pasajeros del segmento. Este incidente generó el extravío de 8 maletas en un solo caso, downgrades no autorizados de Business a Premium, y múltiples cambios de puerta que causaron embarques caóticos. La segunda causa fue el <b>estado deficiente de la flota A350 antigua en la ruta MAD-NRT</b>, donde los sistemas de entretenimiento y wifi resultaron completamente no funcionales, los aseos presentaban mal mantenimiento, y el servicio fue reportado como deficiente. Las métricas operativas confirman este deterioro: la puntualidad cayó 4.67 pts (OTP15 73.53%), el mishandling aumentó 5.16 pts (27.92), y la ocupación alcanzó niveles extremos (Load Factor 95.33%, +1.23 pts). Las rutas más críticas fueron <b>MAD-NRT</b> (NPS –100.0, peor ruta del período), <b>BOG-MAD</b> (NPS –50.0, 4 encuestas), <b>MAD-SCL</b> (NPS –33.3, con quejas de servicio, limpieza de baños y embarque desorganizado), <b>MAD-ORD</b> y <b>MAD-MEX</b> (ambas NPS 0.0). Los pasajeros más afectados fueron los de origen asiático (NPS –100.0, correlacionando con MAD-NRT) y españoles (NPS 0.0, 47% del total, incluyendo afectados por BOG-MAD). Un hallazgo crítico es la diferencia de 36.5 ppts entre la flota A350 antigua (NPS –14.3, 41.2% de las encuestas) y la A350 next (NPS +22.2), evidenciando problemas de mantenimiento y configuración en la flota antigua que requieren atención prioritaria. Este deterioro en Premium LH se canceló con la mejora paradójica de Business LH (+11.9 pts), resultando en un agregado LH falsamente Normal (+1.4 pts) que oculta realidades operativas divergentes.<br>
<br>
En <b>Economy SH</b>, el NPS subió a <b>38.0 (+7.5 pts)</b>, impulsado exclusivamente por <b>YW</b>, que experimentó una mejora excepcional de <b>43.7 (+18.2 pts)</b>. Sin embargo, esta anomalía positiva presenta una <b>contradicción operativa fundamental</b>: el mishandling de YW empeoró significativamente (+7.03 pts hasta 26.74), el Load Factor aumentó (+1.81 pts hasta 80.68%), y la puntualidad se mantuvo estable (OTP15 –0.11 pts). La causa raíz de esta mejora <b>NO ha sido identificada</b> debido a que el análisis de YW está incompleto (solo 1 de 5 herramientas ejecutadas), faltando datos críticos de incidentes operativos, verbatims de clientes, rutas específicas y perfiles de cliente. El nivel de confianza causal es muy bajo. Mientras tanto, <b>IB</b> mantuvo estabilidad con <b>35.1 (+2.1 pts, Normal)</b>, absorbiendo el impacto de 12 cancelaciones por meteorología adversa en LEU, problemas de equipaje documentados en múltiples verbatims, y 28 rutas con incidentes operativos. La magnitud de la mejora de YW fue suficiente para elevar el agregado a anomalía positiva (+7.5 pts), aunque el efecto se diluyó parcialmente por el mayor volumen de IB (346 encuestas, 94% del total vs 22 encuestas de YW, 6% del total). Esta mejora en Economy SH se neutralizó posteriormente con el deterioro de Business SH (–14.2 pts), resultando en un agregado SH falsamente Normal (+5.3 pts).<br>
<br>
En <b>Business LH</b>, el NPS subió a <b>36.4 (+11.9 pts)</b>, pero esta mejora carece de explicación operativa sólida, constituyendo otra <b>anomalía positiva paradójica</b>. Las métricas operativas muestran deterioro: el mishandling aumentó +5.16 pts, la puntualidad cayó 4.67 pts (OTP15 73.53%), la ocupación alcanzó niveles extremos (Load Factor 95.11%, +1.07 pts), y las conexiones perdidas aumentaron +0.38 pts. Los incidentes operativos incluyen el problema técnico del vuelo IB0155 (MAD-BOG) y verbatims negativos documentando equipaje perdido en rutas Brasil-MAD (GRU-MAD con 3 horas de espera, GIG-MAD con equipaje no cargado en clase ejecutiva), retrasos (BOG-MAD con 1 hora de retraso, GRU-MAD con 1.5 horas), y deficiencias de catering en clase ejecutiva (GIG-MAD reportando agotamiento de opciones de pasta). Las rutas con peor desempeño fueron <b>GRU-MAD</b> (NPS –33.3, 3 encuestas), <b>LIM-MAD</b> (NPS 0.0, 2 encuestas), <b>MAD-SJU</b> (NPS 0.0), y <b>GIG-MAD</b> (múltiples verbatims negativos). En contraste, <b>MAD-NRT</b>, <b>BOS-MAD</b> y <b>MAD-MEX</b> registraron NPS 100. Los pasajeros sudamericanos (NPS –16.7, 6 encuestas) y europeos (NPS –25.0, 4 encuestas) fueron los más afectados, mientras que los españoles mostraron alta tolerancia (NPS +66.7, 9 encuestas, 27% de la muestra). Las flotas widebody de largo radio (A350 next NPS 13.3 con 15 encuestas, A332 NPS 16.7) concentraron la insatisfacción, mientras que las operaciones en codeshare con LATAM registraron NPS –100.0. La hipótesis más probable es un sesgo de muestreo (solo 33 encuestas, volumen muy bajo) con sobrerrepresentación de clientes españoles satisfechos que compensaron las rutas problemáticas, o una predisposición positiva por la fecha pre-navideña (21 de diciembre). El nivel de confianza causal es medio-bajo. Esta mejora en Business LH se canceló con el deterioro de Premium LH (–14.7 pts), resultando en un agregado LH falsamente Normal (+1.4 pts).<br>
<br>
La convergencia de <b>Short Haul (Normal, +5.3 pts)</b> y <b>Long Haul (Normal, +1.4 pts)</b>, ambos con variaciones positivas dentro de rangos normales, produjo una acumulación que cruzó el umbral de anomalía a nivel global (+2.8 pts). Sin embargo, esta anomalía positiva global es estadísticamente real pero causalmente inexplicable, resultado de un baseline anormalmente bajo en los 7 días previos o de compensaciones volumétricas donde segmentos Economy (alto volumen) con mejoras sin causa identificada diluyeron los deterioros documentados de Business y Premium (bajo volumen). El sistema presenta cuatro anomalías significativas (dos negativas con causas raíz perfectamente trianguladas: Business SH por meteorología LEU e IB, Premium LH por incidente BOG-MAD y estado de avión; dos positivas sin explicación operativa: Economy SH por YW, Business LH) que se neutralizan mutuamente, generando una falsa sensación de estabilidad en los niveles agregados mientras ocultan volatilidad extrema. Los problemas operativos más graves (IB Business SH –22.8 pts, Premium LH –14.7 pts) solo son visibles en los niveles más desagregados, no en los indicadores de alto nivel.<br>
<br>
<b><u>DETALLE POR CABINA</u></b><br>
<br>
<b><u>Economy SH: Mejora inexplicable impulsada por YW sin correlación operativa</u></b><br>
El segmento registró un <b>NPS de 38.0 (+7.5 pts)</b> en un escenario de transferencia donde <b>YW</b> arrastró a la cabina completa hacia anomalía positiva con <b>43.7 (+18.2 pts)</b>, a pesar de que <b>IB</b> se mantuvo estable con <b>35.1 (+2.1 pts, Normal)</b>. Sin embargo, esta mejora presenta una contradicción operativa crítica: el mishandling de YW empeoró significativamente (+7.03 pts hasta 26.74), el Load Factor aumentó (+1.81 pts hasta 80.68%), y la puntualidad se mantuvo estable (OTP15 –0.11 pts). La causa raíz de esta mejora NO ha sido identificada debido a que el análisis de YW está incompleto (solo 1 de 5 herramientas ejecutadas), faltando datos críticos de incidentes operativos, verbatims de clientes, rutas específicas y perfiles de cliente. El nivel de confianza causal es muy bajo. IB, por su parte, absorbió el impacto de 12 cancelaciones por meteorología adversa en LEU, problemas de equipaje documentados en múltiples verbatims, y 28 rutas con incidentes operativos, manteniendo estabilidad sin generar anomalía. La magnitud de la mejora de YW fue suficiente para elevar el agregado a +7.5 pts, aunque el efecto se diluyó parcialmente por el mayor volumen de IB (346 encuestas, 94% del total vs 22 encuestas de YW, 6% del total). Esta anomalía positiva requiere validación urgente mediante la completitud del análisis de YW para identificar si la mejora es real o un artefacto estadístico.<br>
<br>
<b><u>Business SH: Colapso operativo de IB por meteorología adversa domina el agregado</u></b><br>
El segmento registró un <b>NPS de 17.8 (–14.2 pts)</b> en un escenario de dominancia donde <b>IB</b> impuso su caída crítica de <b>16.1 (–22.8 pts)</b> sobre el agregado, a pesar de que <b>YW</b> mantuvo estabilidad con <b>21.4 (+5.4 pts, Normal)</b>. El colapso de IB se originó en la meteorología adversa en el aeropuerto de Lleida-Alguaire (LEU), que generó 12 cancelaciones masivas (92% de los incidentes del día), desencadenando un efecto cascada en la red doméstica española. Este evento deterioró la puntualidad (OTP15 cayó 4.48 pts hasta 87.13%) y disparó los problemas de equipaje (Mishandling aumentó 4.54 pts hasta 28.29), casi duplicando la tasa normal. Las conexiones perdidas también aumentaron (Misconex +0.47 pts hasta 1.37). Las rutas más afectadas fueron GVA-MAD (NPS 0.0, 3 pasajeros reportando retrasos de 45-50 minutos en entrega de equipaje sin respetar prioridades Business), EAS-MAD (NPS 0.0, con quejas críticas sobre tripulación tardía y pasajeros esperando 20 minutos al frío), CMN-MAD (NPS 0.0, equipaje no entregado incluyendo medicación y documentos valiosos), BLQ-MAD (NPS 0.0, cancelación sin compensación con esperas superiores a 5 horas), DUS-MAD y MAD-VGO (ambas NPS 0.0). Los viajeros de negocios fueron los más sensibles (NPS –50.0 vs Leisure 21.4), especialmente aquellos de origen europeo no español (NPS 5.6) y sudamericano (NPS –25.0). La flota A321 mostró el peor rendimiento (NPS –11.1) en operaciones de corto radio. El volumen de IB (41 encuestas, 91% del total) aplastó el efecto neutral de YW (4 encuestas, 9%), generando una anomalía negativa agregada que suaviza parcialmente (–14.2 pts) la caída crítica de IB (–22.8 pts). La diferencia de 44.2 pts entre ambas compañías evidencia la magnitud del problema operativo.<br>
<br>
<b><u>Economy LH: Estabilidad operativa sin cambios significativos</u></b><br>
El segmento mantuvo un <b>NPS de 3.7 (+0.2 pts, Normal)</b> sin cambios significativos detectados. La variación se mantuvo dentro del rango normal de fluctuación (menor a 3 pts), actuando como estabilizador dentro del radio Long Haul a pesar del contexto operativo adverso general. No se identificaron desviaciones operativas críticas específicas para este segmento, y no hubo rutas o perfiles con reactividad significativa que reportar.<br>
<br>
<b><u>Business LH: Mejora paradójica sin causa operativa identificada</u></b><br>
El segmento registró un <b>NPS de 36.4 (+11.9 pts)</b>, pero esta mejora carece de explicación operativa sólida, constituyendo una anomalía positiva paradójica. Las métricas operativas muestran deterioro: el mishandling aumentó +5.16 pts, la puntualidad cayó 4.67 pts (OTP15 73.53%), la ocupación alcanzó niveles extremos (Load Factor 95.11%, +1.07 pts), y las conexiones perdidas aumentaron +0.38 pts. Los incidentes operativos incluyen el problema técnico del vuelo IB0155 (MAD-BOG) y verbatims negativos documentando equipaje perdido en rutas Brasil-MAD (GRU-MAD con 3 horas de espera, GIG-MAD con equipaje no cargado en clase ejecutiva), retrasos (BOG-MAD con 1 hora de retraso, GRU-MAD con 1.5 horas), y deficiencias de catering en clase ejecutiva (GIG-MAD reportando agotamiento de opciones de pasta). Las rutas con peor desempeño fueron GRU-MAD (NPS –33.3, 3 encuestas), LIM-MAD (NPS 0.0, 2 encuestas), MAD-SJU (NPS 0.0), y GIG-MAD (múltiples verbatims negativos). En contraste, MAD-NRT, BOS-MAD y MAD-MEX registraron NPS 100, sugiriendo una concentración de respuestas positivas en rutas no afectadas. Los pasajeros sudamericanos (NPS –16.7, 6 encuestas) y europeos (NPS –25.0, 4 encuestas) fueron los más afectados, mientras que los españoles mostraron alta tolerancia (NPS +66.7, 9 encuestas, 27% de la muestra). Las flotas widebody de largo radio (A350 next NPS 13.3 con 15 encuestas, A332 NPS 16.7) concentraron la insatisfacción, mientras que las operaciones en codeshare con LATAM registraron NPS –100.0. La hipótesis más probable es un sesgo de muestreo (solo 33 encuestas, volumen muy bajo) con sobrerrepresentación de clientes españoles satisfechos que compensaron las rutas problemáticas, o una predisposición positiva por la fecha pre-navideña (21 de diciembre). El nivel de confianza causal es medio-bajo, y se requiere análisis con mayor volumen de datos para confirmar o descartar esta hipótesis.<br>
<br>
<b><u>Premium LH: Deterioro por incidente BOG-MAD y estado deficiente de flota A350</u></b><br>
El segmento registró un <b>NPS de 5.9 (–14.7 pts)</b> debido a dos causas raíz convergentes con alta confianza. La primera fue el incidente técnico en el vuelo IB0155 (MAD-BOG del 21 de diciembre), que resultó en una reprogramación de 24 horas (vuelo trasladado al IB157 del 22 de diciembre), afectando al 23.5% de los pasajeros del segmento. Este incidente generó el extravío de 8 maletas en un solo caso, downgrades no autorizados de Business a Premium, y múltiples cambios de puerta que causaron embarques caóticos. La segunda causa fue el estado deficiente de la flota A350 antigua en la ruta MAD-NRT, donde los sistemas de entretenimiento y wifi resultaron completamente no funcionales, los aseos presentaban mal mantenimiento, y el servicio fue reportado como deficiente. Las métricas operativas confirman este deterioro: la puntualidad cayó 4.67 pts (OTP15 73.53%), el mishandling aumentó 5.16 pts (27.92, incremento del 22%), y la ocupación alcanzó niveles extremos (Load Factor 95.33%, +1.23 pts). Las rutas más críticas fueron MAD-NRT (NPS –100.0, peor ruta del período), BOG-MAD (NPS –50.0, 4 encuestas), MAD-SCL (NPS –33.3, con quejas de servicio, limpieza de baños y embarque desorganizado), MAD-ORD y MAD-MEX (ambas NPS 0.0). En contraste, LIM-MAD y MAD-SJO registraron NPS +100.0. Los pasajeros más afectados fueron los de origen asiático (NPS –100.0, correlacionando con MAD-NRT) y españoles (NPS 0.0, 47% del total, incluyendo afectados por BOG-MAD). Un hallazgo crítico es la diferencia de 36.5 ppts entre la flota A350 antigua (NPS –14.3, 41.2% de las encuestas) y la A350 next (NPS +22.2), evidenciando problemas de mantenimiento y configuración en la flota antigua que requieren atención prioritaria.
```
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-20 to 2025-12-20:
❌ Error en la interpretación jerárquica: An error occurred (ExpiredTokenException) when calling the Converse operation: The security token included in the request is expired
🚨 Anomalías detectadas: daily_analysis

📅 2025-12-19 to 2025-12-19:
❌ Error en la interpretación jerárquica: An error occurred (ExpiredTokenException) when calling the Converse operation: The security token included in the request is expired
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
