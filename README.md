## NPS Aggregation Tree – detección de anomalías (+/-/N)

```markdown
Global: +/-/N
│
├── LH (Long Haul)
│   └── +/-/N
│       ├── Economy
│       │   └── +/-/N
│       ├── Business
│       │   └── +/-/N
│       └── Premium
│           └── +/-/N
│
└── SH (Short Haul)
    └── +/-/N
        ├── Economy
        │   └── +/-/N
        │       ├── IB: +/-/N
        │       └── YW: +/-/N
        └── Business
            └── +/-/N
                ├── IB: +/-/N
                └── YW: +/-/N
```

---

## Contexto temporal y tipos de análisis

### Análisis Mensual
- **Propósito**: Evaluar el rendimiento del mes en curso y compararlo con objetivos mensuales.
- **Fuente de datos**: Valores depurados mensuales de las tablas de cabina.
- **Contexto**: 
  - Si hay ≥10 días de mes: se evalúa el mes actual.
  - Si hay <10 días de mes: se usa el mes anterior como contexto (en el primer lunes tras el cierre).

### Análisis Diario
- **Propósito**: Monitorear tendencias semanales y detectar desviaciones significativas.
- **Fuente de datos**: Valores diarios de evolución.
- **Contexto**: 
  - Se compara cada día con su media móvil de 7 días (ventana centrada).
  - Para Global: también se compara con el objetivo mensual.
  - Si hay <10 días de mes: se usa el mes anterior como referencia para objetivos.

---

## Definición de anomalía por nodo (+/-/N)

### Global

* **+** si el NPS mensual actual supera el target en +2 puntos o más y llevamos ≥10 días del mes.
* **-** si el NPS mensual actual está por debajo del target en -2 puntos o más y llevamos ≥10 días del mes.
* **+/-** en el primer lunes tras el cierre natural del mes o del proceso de ponderación de pesos, si el NPS del mes anterior supera/difiere del target en ≥2 puntos.
* **N** en cualquier otro caso.

### LH (Long Haul)

* **+** si el NPS de LH en el periodo supera el target en ≥2 puntos.
* **-** si el NPS de LH en el periodo está por debajo del target en ≥2 puntos.
* **+/-** en cierre si cada cabina (Economy o Premium) se sale de ±2 puntos.
* **N** si la desviación global de LH es <2 puntos y ambas cabinas están dentro de ±2.

### SH (Short Haul)

* **+** si el NPS de SH supera el target en ≥2 puntos.
* **-** si el NPS de SH está por debajo del target en ≥2 puntos.
* **N** si la desviación está dentro de ±2 puntos.

#### SH → Economy

* **+** si el NPS supera el target en ≥2 puntos.
* **-** si el NPS está por debajo del target en ≥2 puntos.
* **+/-** en cierre de mes si esta cabina (70% del peso) sale de ±2.
* **N** en caso contrario.

#### SH → Business

* **+** si el NPS supera el target en ≥2 puntos.
* **-** si el NPS está por debajo del target en ≥2 puntos.
* **N** en caso contrario.

##### IB (dentro de SH → Business)

* **+** si el NPS supera el target en ≥2 puntos.
* **-** si el NPS está por debajo del target en ≥2 puntos.
* **N** en caso contrario.

##### YW (dentro de SH → Business)

* **+** si el NPS supera el target en ≥2 puntos.
* **-** si el NPS está por debajo del target en ≥2 puntos.
* **N** en caso contrario.

### Detección de anomalías diarias

* **Propósito**: Detectar desviaciones significativas en el contexto semanal.
* **Método**: 
  - Calcular la media los ultimos 7 días.
  - **+** si el valor diario supera la media en +10 puntos o más.
  - **-** si el valor diario está por debajo de la media en -10 puntos o más.
  - Para Global: también se compara con el objetivo mensual (±2 puntos).
* **Contexto temporal**:
  - Si hay ≥10 días de mes: se usa el objetivo del mes actual.
  - Si hay <10 días de mes: se usa el objetivo del mes anterior.

### Ajuste por fase del mes

* **Mes en curso (<10 días):** 
  - Solo Global y el nodo de cabina principal (LH o SH → Economy) se evalúan provisionalmente.
  - Se usa el mes anterior como contexto para objetivos.
  - El resto queda `N` hasta cierre.
* **Cierre de mes (≥10 días):** 
  - Se evalúan todos los nodos y subnodos.
  - Se usa el mes actual como contexto para objetivos.

---

## Algoritmo de interpretación de anomalías (bottom-up)

El análisis se realiza de abajo hacia arriba, evaluando primero los nodos hijos y su relación con el padre. Para cada nodo padre, se consideran las siguientes situaciones:

### Casos homogéneos (todos los hijos con el mismo estado)

1. **Todos los hijos son N**
   * Si el padre es N: Normal, no hay anomalías.
   * Si el padre es + o -: Inconsistencia detectada. Revisar posibles errores en la ponderación o en la detección de anomalías.

2. **Todos los hijos son +**
   * Si el padre es +: Anomalía positiva consistente. La causa raíz afecta a todos los hijos.
   * Si el padre es - o N: Inconsistencia detectada. Revisar posibles errores en la ponderación o en la detección de anomalías.

3. **Todos los hijos son -**
   * Si el padre es -: Anomalía negativa consistente. La causa raíz afecta a todos los hijos.
   * Si el padre es + o N: Inconsistencia detectada. Revisar posibles errores en la ponderación o en la detección de anomalías.

### Casos de mezcla

4. **Mezcla de + y - (sin N)**
   * Si el padre es N: Las anomalías se cancelan entre sí en la ponderación.
   * Si el padre es +: La anomalía positiva tiene mayor peso en la ponderación.
   * Si el padre es -: La anomalía negativa tiene mayor peso en la ponderación.

5. **Mezcla de N y + (sin -)**
   * Si el padre es N: La anomalía positiva se diluye en la ponderación por el peso de los nodos N.
   * Si el padre es +: La anomalía positiva es suficientemente significativa para afectar al padre.
   * Si el padre es -: Inconsistencia detectada. Revisar posibles errores.

6. **Mezcla de N y - (sin +)**
   * Si el padre es N: La anomalía negativa se diluye en la ponderación por el peso de los nodos N.
   * Si el padre es -: La anomalía negativa es suficientemente significativa para afectar al padre.
   * Si el padre es +: Inconsistencia detectada. Revisar posibles errores.

7. **Mezcla de N, + y -**
   * Si el padre es N: Puede ser una combinación de dilución y cancelación de anomalías.
   * Si el padre es +: La anomalía positiva tiene mayor peso en la ponderación.
   * Si el padre es -: La anomalía negativa tiene mayor peso en la ponderación.

### Interpretación detallada

Para cada caso, la interpretación debe incluir:
1. Naturaleza de los nodos hijos (cabina, ruta, etc.)
2. Naturaleza del nodo padre
3. Reflexión sobre la relación entre los estados
4. En caso de anomalías, indicar qué nodos son los principales causantes según la ponderación

Ejemplo de interpretación:
```
SH Business (padre: +)
├── IB (hijo: +)
└── YW (hijo: N)

Interpretación: "En SH Business (padre), la anomalía positiva se debe principalmente a IB, 
mientras que YW se mantiene en valores normales. La anomalía de IB es suficientemente 
significativa para afectar al padre a pesar de la dilución por YW."
```

---

## Fuentes de causalidad

### Fuentes post-encuesta (perceptivas)

* **Explanatory Drivers**: puntuaciones de otras preguntas de la encuesta.
* **Verbatims**: respuestas en lenguaje natural.

### Fuentes objetivas (causas reales)

* **Operative**: OTP, mishandling, missconnections.
* **NCS**: incidencias operativas registradas.

Relación percepción–realidad: las perceptivas muestran cómo se sintió el cliente; las objetivas, qué ocurrió realmente.

---

## Análisis de fuentes de causalidad

### Operative

Para explicar anomalías de NPS con datos operativos:

1. Identificar días anómalos en puntualidad (OTP) comparando cada día con la media semanal y con el objetivo.
2. Confrontar esas fechas con las anomalías diarias de NPS.
3. Independientemente de la coincidencia, anotar ambos y analizarlos junto a otras fuentes.

### Explanatory Drivers

*(Por completar; metodología de uso de puntuaciones de encuesta.)*

### Verbatims

*(Por completar; análisis de comentarios en texto libre.)*

### NCS

*(Por completar; uso del registro de incidencias.)*

---

## Interpretación del grafo causal de anomalías

1. **Ejemplo clásico de temporada alta**
   En verano, a pesar de un OTP bueno, detectamos caídas de NPS.

   * Causa raíz: *load factor* provoca más colas y molestias.
   * Se refleja en Explanatory Drivers y Verbatims, no en Operative.
   * En el grafo: Global y cabinas con `+` o `-`, explicadas por variables perceptivas.

2. **Ejemplo de incidencias Operative**
   Eventos que relacionamos con caídas de NPS en el grafo:

   * Mishandling en Bogotá y Ciudad de México (alta altitud/calor) donde se dejan maletas por límite de peso.
   * Huelgas de controladores en París y Alemania que, por saturación del ATC europeo, afectan múltiples aeropuertos.
   * Fallos del SATE en Barajas, paralizando el transporte automático de equipajes.

3. *(Más escenarios por añadir)*


** Adicionalmente en Explanatory Drivers, ademas del research de subgrupos anómalos, cabina radio vemos vs 14 dias y vs targets.
** Asi vemos la evolución, no solo aislado con respecto a los últimos 7 días, y nos da para hacer una comparativa.

### Lógica de explicación de anomalías

El proceso de explicación de anomalías se realiza generación por generación, de abajo hacia arriba en el árbol. Para cada nivel, se analiza la relación padre-hijos de la siguiente manera:

#### Determinación del nivel de la anomalía

1. **Para casos de mezcla (+/-/N)**
   * La anomalía siempre está a nivel de los hijos individuales
   * Cada hijo con anomalía requiere su propia explicación
   * El estado del padre es resultado de la agregación ponderada

2. **Para casos homogéneos (todos + o todos -)**
   * Se requiere análisis individual de cada hijo usando el agente explicador de anomalías
   * Si todas las explicaciones son idénticas: la anomalía es a nivel padre
   * Si las explicaciones son diferentes: son anomalías individuales que coinciden en signo

#### Proceso de construcción del prompt

1. **Análisis generacional**
   * Comenzar desde los nodos hoja
   * Para cada nivel:
     - Analizar la relación padre-hijos
     - Generar explicaciones para cada anomalía
     - Determinar si la explicación es a nivel padre o de hijos
     - Guardar las explicaciones generadas

2. **Construcción del prompt**
   * Para cada nivel analizado:
     ```
     Nivel actual: [nivel]
     Estado padre: [estado]
     Estados hijos: [estados]
     
     Explicaciones individuales:
     - Hijo 1: [explicación]
     - Hijo 2: [explicación]
     ...
     
     Análisis de agregación:
     [Determinación si es anomalía padre o agregación]
     
     Explicación final:
     [Explicación consolidada para este nivel]
     ```

3. **Propagación hacia arriba**
   * Las explicaciones generadas se convierten en contexto para el siguiente nivel
   * El prompt se enriquece con cada nivel analizado
   * Se mantiene la trazabilidad de las explicaciones

Ejemplo de proceso:
```
Nivel 1 (Hojas):
SH Business
├── IB (hijo: +) → Explicación: "Problemas operativos en Barajas"
└── YW (hijo: +) → Explicación: "Problemas operativos en Barajas"

Análisis: Explicaciones idénticas → Anomalía a nivel padre
Prompt generado: "Anomalía positiva en SH Business debido a problemas operativos en Barajas"

Nivel 2 (Subiendo):
SH
├── Business (hijo: +) → Explicación: "Problemas operativos en Barajas"
└── Economy (hijo: N)

Análisis: Mezcla de estados → Anomalía a nivel hijo
Prompt actualizado: "En SH, la anomalía positiva en Business se debe a problemas operativos en Barajas, 
mientras que Economy se mantiene en valores normales. La anomalía se diluye en el padre debido al peso de Economy."
```

Este proceso asegura que:
1. Cada nivel se analiza de forma independiente
2. Las explicaciones se construyen de manera incremental
3. Se mantiene la trazabilidad de las causas
4. Se distingue claramente entre anomalías a nivel padre y agregaciones de anomalías de hijos



Si saltan puntuality, aconnections, load factor mirar operative. Con arrivals mriar los dos operative y verbatims. Y con los demas verbatims. 

## Requisitos de configuración

### Autenticación Azure AD

Para acceder a los datos de Power BI, se requiere una configuración correcta de Azure AD:

1. **Credenciales necesarias**:
   ```python
   tenant_id = "your_tenant_id"        # GUID o nombre del tenant
   client_id = "your_client_id"        # ID de la aplicación registrada
   client_secret = "your_secret"       # Secreto de la aplicación
   ```

2. **Formato del tenant**:
   * Debe ser un GUID válido o nombre de tenant
   * URL base: `https://login.microsoftonline.com/{tenant_id}`
   * No puede ser "None" o vacío

3. **Registro de aplicación**:
   * La aplicación debe estar registrada en Azure AD
   * Debe tener los permisos necesarios para Power BI
   * Scope requerido: `https://analysis.windows.net/powerbi/api/.default`

4. **Manejo de errores comunes**:
   * Error 90002: Tenant no encontrado
     - Verificar que el tenant_id es correcto
     - Confirmar que hay suscripciones activas
     - Comprobar que se está usando la nube correcta
   * Error de OIDC Discovery:
     - Verificar el formato de la URL del tenant
     - Comprobar la conectividad con Azure AD
     - Validar que el tenant existe y está activo

5. **Variables de entorno recomendadas**:
   ```bash
   AZURE_TENANT_ID=your_tenant_id
   AZURE_CLIENT_ID=your_client_id
   AZURE_CLIENT_SECRET=your_client_secret
   ```