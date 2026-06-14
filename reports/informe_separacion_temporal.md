# Informe Técnico — Separación Temporal de Datos Clínicos  
## Predicción de Mortalidad y Necesidad de UCI Postquirúrgica  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Etapa:** Preprocesamiento — Construcción del conjunto de datos  
**Autor:** Iker Arias  
**Fecha:** Abril 2026  

---

## Resumen Ejecutivo

Se describe el diseño, implementación y verificación del pipeline de separación temporal aplicado sobre 46 archivos CSV de origen hospitalario (5,144,115 filas en total). El objetivo es segmentar cada registro clínico en función de su relación temporal con el evento quirúrgico de referencia de cada paciente, produciendo conjuntos de datos **pre-operatorios**, **intra-operatorios**, **post-operatorios**, **estáticos** y **sin referencia** que sirvan de base para el modelado predictivo.

El pipeline ha sido auditado exhaustivamente: se verificaron 4,294,675 filas con marca temporal contra la ventana quirúrgica de referencia obteniendo **0 violaciones temporales**, mutua exclusividad perfecta entre segmentos y trazabilidad completa de todos los archivos.

---

## 1. Introducción y Motivación

### 1.1 Contexto clínico

El conjunto de datos proviene del sistema de información clínica del hospital y cubre a pacientes sometidos a cirugía electiva. Cada paciente genera registros en múltiples subsistemas (laboratorio, farmacia, anestesia, microbiología, radiología…), dispersos a lo largo del tiempo. Para construir un modelo predictivo de mortalidad y necesidad de UCI postquirúrgica, es imprescindible separar estos registros en función de su relación con la cirugía:

- **Pre-operatorio**: datos de evaluación y seguimiento previos a la cirugía — resumen el estado basal del paciente.  
- **Intra-operatorio**: registros ocurridos durante la intervención — capturan la dinámica del acto quirúrgico.  
- **Post-operatorio**: registros posteriores — constituyen el contexto de evolución y son la fuente de las variables objetivo.

Una asignación incorrecta de cualquier registro (ej. que datos post-op contaminen el pre-op) introduciría **data leakage** que invalidaría completamente los modelos de predicción.

### 1.2 Fuentes de datos

| Tipo | Archivos | Descripción |
|---|---|---|
| Referencia quirúrgica | `009_INTERVENCIONES_QUIRURGICAS_INTERVENCION.csv` | Define la ventana temporal de cada cirugía |
| Archivos temporales | 37 archivos CSV | Registros con marca de tiempo (`fecha_evento`) |
| Archivos estáticos | 9 archivos CSV | Sin dimensión temporal; características fijas del paciente |

---

## 2. Diseño del Pipeline

### 2.1 Arquitectura general

El pipeline se estructura en tres fases secuenciales:

```
data/data_raw/        →  FASE 1: Normalización  →  data/data_normalized/
                         (preprocces.py)

data/data_normalized/ →  FASE 2: Segmentación   →  data/data_divided/
                         (divide.py)                ├── pre/
                                                    ├── intra/
                                                    ├── post/
                                                    ├── static/
                                                    └── sin_referencia/
```

El código fuente se organiza en el paquete `src/`:

```
src/
├── preproccesing/
│   ├── preprocces.py     # Fase 1: normalización de columnas
│   └── divide.py         # Fase 2: segmentación temporal
└── utils/
    ├── vision_data.py    # Carga, detección de fechas y selección de episodio
    └── division_ev.py    # Análisis y visualización de resultados
```

### 2.2 Fase 1 — Normalización de columnas (`preprocces.py`)

**Problema**: los 46 archivos CSV provienen de subsistemas distintos del hospital y usan nombres de columna heterogéneos para referirse al mismo concepto. Por ejemplo, el identificador de paciente aparece como `Id Paciente`, `Id_Paciente`, `Identificador de Paciente` o `idpaci` dependiendo del módulo.

**Solución**: se aplica un mapping de prioridad sobre listas de patrones conocidos para renombrar todas las variantes a un nombre canónico:

| Columna canónica | Variantes cubiertas | Verificación empírica |
|---|---|---|
| `id_paciente` | `Id Paciente`, `Id_Paciente`, `Identificador de Paciente`, `idpaci` | ✅ Mismo valor numérico para el mismo paciente en todos los archivos |
| `fecha_evento` | `Hora (inicio)`, `Fecha (administración)`, `Fecha Dispensación`, `Fecha (solicitud)`, `Fecha (registro)`, `Hora (form cte)`, `Hora (ingreso)`, `Fecha DBP`, `Hora (realización proc)`, `Fecha Hora (inicio)`, `Día (Fecha)`, `Fecha (alta)`, `Fecha (ingreso)` (13 variantes activas) | ✅ Todas contienen timestamps del momento en que ocurrió el evento clínico |
| `id_episodio` | `Episodio (único)`, `Episodio (único) INTER`, `Episodio (único) SOLIC` | ⚠️ Ver nota |

> **Nota sobre `id_episodio`**: las tres variantes no son el mismo concepto. Para el mismo paciente y misma cirugía, `Episodio (único) SOLIC` (~57M) corresponde al episodio de la *solicitud quirúrgica*, `Episodio (único) INTER` (~62M) al episodio del *ingreso hospitalario* donde se realizó la intervención, y `Episodio (único)` genérico al episodio de hospitalización. Son identificadores administrativos distintos dentro del workflow del hospital. **Esto no afecta a la segmentación temporal**, que únicamente utiliza `id_paciente` como clave de join. `id_episodio` se normaliza como metadato y no interviene en ninguna operación de segmentación (`divide.py` no lo referencia).

**Caso especial — archivo 008**: `008_INTERVENCIONES_QUIRURGICAS_CONTACTO_ANESTESIA.csv` no contiene columna de paciente en su formato original (solo `cod Solicitud`). Se recupera `id_paciente` mediante un *join* con `009_INTERVENCIONES_QUIRURGICAS_INTERVENCION.csv` vía `cod Solicitud`, logrando un 100% de correspondencia (177 filas).

**Archivos estáticos**: 9 archivos sin dimensión temporal (diagnósticos CIE, comorbilidades Charlson, checklist de cirugía) se copian directamente a `data_normalized/` sin modificación de contenido.

**Resultado**: 46/46 archivos normalizados, 37/37 archivos no-estáticos con `id_paciente` + `fecha_evento`.

### 2.3 Fase 2 — Selección del episodio de referencia (`vision_data.py`)

Dado que un mismo paciente puede tener múltiples cirugías en el historial, se selecciona una única intervención de referencia por paciente siguiendo una regla clínicamente motivada:

1. Se filtran únicamente cirugías **principales** (campo `Principal o secundaria == "PRINCIPAL"`).  
2. Entre ellas, se prioriza **CIRUGÍA MAYOR** sobre CIRUGÍA MENOR.  
3. Dentro de cada tipo, se selecciona la **más antigua cronológicamente** (primer `Hora (inicio)`).

Esta selección produce exactamente **12,288 pacientes únicos**, cada uno con una ventana quirúrgica $[t_{\text{inicio}}, t_{\text{fin}}]$ definida.

### 2.4 Fase 2 — Segmentación temporal (`divide.py`)

La función principal `segmentar_y_guardar()` procesa cada archivo normalizado y aplica la siguiente lógica:

**Para archivos estáticos (9 archivos)**:

$$\text{archivo} \rightarrow \texttt{/static/}$$

**Para el resto (37 archivos)**:

1. Se carga el archivo y se convierte `fecha_evento` a `datetime`.
2. Se hace un `LEFT JOIN` con la referencia sobre `id_paciente`.
3. Las filas **sin match** (pacientes no en la referencia) van a `sin_referencia/`.
4. Las filas **con match** se clasifican mediante la función `determinar_tramo()`:

$$\text{PRE}: \quad fecha\_evento < t_{\text{inicio}}$$

$$\text{INTRA}: \quad t_{\text{inicio}} \leq fecha\_evento \leq t_{\text{fin}}$$

$$\text{POST}: \quad fecha\_evento > t_{\text{fin}}$$

Este criterio es **por comparación directa de timestamps** con precisión de minuto, sin ventanas de tolerancia ni aproximaciones. Un registro pertenece a exactamente uno de los tres conjuntos — las condiciones son mutuamente excluyentes y exhaustivas para cualquier `fecha_evento` no nula.

---

## 3. Correcciones de Bugs Detectados en la Auditoría

Durante la auditoría exhaustiva del código y los datos, se identificaron y corrigieron 4 bugs que habrían comprometido la integridad de los datos de haberse dejado sin resolver.

### Bug 1 — Archivos estáticos sin clasificar (preprocces.py + divide.py)

**Descripción**: los archivos `001_INTERVENCIONES_QUIRURGICAS.csv`, `004_DIAGNOSTICO_PRINCIPAL_CIE10.csv` y `005_DIAGNOSTICO_PRINCIPAL_CIE9.csv` no estaban incluidos en la lista `ARCHIVOS_ESTATICOS`. Como no tienen columna `fecha_evento`, el pipeline intentaba procesarlos como temporales y los ignoraba silenciosamente.

**Impacto**: 3 archivos con variables diagnósticas clave (diagnóstico principal CIE-9, CIE-10 y codificación de la intervención) ausentes del dataset final.

**Corrección**: añadidos a `ARCHIVOS_ESTATICOS` en ambos módulos → `static/` pasó de 6 a 9 archivos.

### Bug 2 — Archivo 008 sin identificador de paciente (preprocces.py)

**Descripción**: `008_INTERVENCIONES_QUIRURGICAS_CONTACTO_ANESTESIA.csv` almacena únicamente `cod Solicitud` como identificador, sin columna `id_paciente`. Al no encontrar la columna canónica, el archivo era ignorado silenciosamente.

**Impacto**: 177 registros de contacto con anestesia — información clínica relevante para el preoperatorio — completamente perdidos.

**Corrección**: caso especial en `preprocces.py` que recupera `id_paciente` mediante join con 009 vía `cod Solicitud` antes de la normalización (coincidencia 100%).

### Bug 3 — Pérdida total del archivo 009 por colisión de columnas (divide.py)

**Descripción**: el archivo `009_INTERVENCIONES_QUIRURGICAS_INTERVENCION.csv`, que es también el archivo de referencia quirúrgica, contiene en su forma normalizada la columna `Hora Intervención (fin)`. Al hacer el `LEFT JOIN` con la tabla de referencia (que también tiene esa columna), pandas generaba sufijos `_x` / `_y` automáticamente. La función `determinar_tramo()` buscaba `Hora Intervención (fin)` (sin sufijo) → `KeyError` → capturado por `except Exception` → **12,288 registros quirúrgicos completamente perdidos en silencio**.

**Impacto**: el segmento `intra/` carecía del archivo más importante para el modelo — los registros de la propia intervención. 0 de 12,288 filas eran procesadas.

**Corrección**: en `segmentar_y_guardar()`, antes del merge, se eliminan del DataFrame de entrada las columnas que colisionarían con las de referencia:

```python
cols_ref = ["Hora (inicio)", "Hora Intervención (fin)"]
cols_colision = [c for c in cols_ref if c in df.columns]
if cols_colision:
    df = df.drop(columns=cols_colision)
```

**Resultado**: `intra/` pasó de 19 a 20 archivos; `009` aparece con 15,086 filas.

### Bug 4 — Horas de inicio y fin invertidas para 2 pacientes (divide.py)

**Descripción**: en los datos del hospital, 2 pacientes (IDs 44012 y 73006) tenían registrada la `Hora (inicio)` mayor que `Hora Intervención (fin)` (verosímilmente un error de transcripción en el sistema hospitalario). Con la ventana invertida, cualquier registro del paciente en esa franja temporal aparecería simultáneamente en pre Y en post (ya que la condición de pre es `fecha < inicio` y la de post es `fecha > fin`, y con `inicio > fin` ambas son satisfacibles por la misma fecha).

**Impacto**: data leakage — las mismas filas presentes en pre y post para esos 2 pacientes. Afecta a paciente 44012 (dur. invertida = -0.83h) y 73006 (dur. invertida = -9.08h).

**Corrección**: en `normalizar_ref()`, detección y corrección automática:

```python
mask_inv = df_ref["Hora (inicio)"] > df_ref["Hora Intervención (fin)"]
df_ref.loc[mask_inv, ["Hora (inicio)", "Hora Intervención (fin)"]] = \
    df_ref.loc[mask_inv, ["Hora Intervención (fin)", "Hora (inicio)"]].values
```

**Resultado verificado**: paciente 44012 → ventana corregida [10:10, 11:00] (0.83h); paciente 73006 → [10:15, 19:20] (9.08h). Verificación en Bloque 2 confirma 0 filas idénticas cross-segment.

---

## 4. Verificación Formal del Pipeline

Se diseñó y ejecutó una batería de 9 checks exhaustivos sobre la totalidad de los datos (sin muestreo).

### 4.1 Metodología de verificación

| Bloque | Check | Método |
|---|---|---|
| 0 | Integridad de la referencia | Estadísticas de duración quirúrgica, detección de invertidos |
| 1A | Cobertura raw → normalized | Comparación de conjuntos de nombres de archivo |
| 1B | Columnas canónicas | Lectura de cabecera de 37 archivos no-estáticos |
| 1C | 009 en intra/ | Existencia y conteo de filas |
| 1D | Archivos estáticos | Verificación del conjunto exacto en static/ |
| 2A | Mutua exclusividad | pre+intra+post ≤ normalized para cada archivo |
| 2B | Pacientes invertidos | Ventana temporal de los 2 pacientes corregidos |
| 3 | Integridad temporal | **Todas** las filas de pre, intra y post comparadas contra ref |
| 4 | Cobertura de pacientes | Cardinalidad de intersección por segmento |
| 5 | Trazabilidad duplicados | Cada duplicado en dividido existe también en raw |

### 4.2 Resultados

| Check | Filas verificadas | Resultado |
|---|---|---|
| raw = normalized | 46 archivos | ✅ 100% |
| columnas canónicas | 37 archivos | ✅ 100% |
| 009 en intra/ | 15,086 filas | ✅ presente |
| Mutua exclusividad | 37 archivos | ✅ 0 archivos con overflow |
| Integridad PRE | **2,739,540 filas** | ✅ 0 violaciones |
| Integridad INTRA | **38,683 filas** | ✅ 0 violaciones |
| Integridad POST | **1,516,452 filas** | ✅ 0 violaciones |
| Cobertura pre | 12,288/12,288 | ✅ 100.00% |
| Cobertura intra | 12,288/12,288 | ✅ 100.00% |
| Cobertura post | 12,075/12,288 | ✅ 98.27% (*) |
| sin_ref ∩ ref | — | ✅ ∅ |
| Duplicados pipeline | 4 archivos | ✅ origen hospital |

**(*) 213 pacientes (1.73%) sin registros post-operatorios**: interpretado como alta médica sin registros posteriores capturados en el sistema, o traslado a otro centro. No constituye un error del pipeline dado que los datos correspondientes están ausentes también en raw.

**Avisos no críticos**:
- 3 cirugías con duración = 0 minutos (inicio == fin): probable registro al mismo minuto por redondeo; los registros son clasificables sin ambigüedad.
- 116 filas duplicadas en archivos `010_LABORATORIO*`: originadas en el raw del hospital (misma redundancia presente en los archivos brutos).

---

## 5. Resultados: Estructura Final del Dataset

### 5.1 Distribución por segmento

| Segmento | Archivos | Filas | Descripción |
|---|---|---|---|
| **pre** | 29 | 2,739,540 | Eventos anteriores a la cirugía |
| **intra** | 20 | 38,683 | Eventos durante la intervención |
| **post** | 32 | 1,516,452 | Eventos posteriores a la cirugía |
| **static** | 9 | 213,350 | Datos sin dimensión temporal |
| **sin_referencia** | 34 | 636,090 | Pacientes sin cirugía de referencia |
| **TOTAL** | — | **5,144,115** | — |

### 5.2 Estadísticas de la referencia quirúrgica

| Métrica | Valor |
|---|---|
| Pacientes con cirugía de referencia | 12,288 |
| Duración media de cirugía | 1.28 h |
| Duración mediana | 0.92 h |
| Duración mínima | 0.00 h (3 casos) |
| Duración máxima | 14.42 h |
| Pacientes con horas invertidas (corregidos) | 2 |

### 5.3 Cobertura por subsistema

Todos los 37 archivos con dimensión temporal del hospital están presentes en al menos uno de los segmentos temporales (pre/intra/post). El archivo de referencia quirúrgica (`009`) está correctamente representado en `intra/` con 15,086 filas — el número completo de registros del episodio principal de cada paciente.

---

## 6. Discusión sobre la Validez del Dataset

### 6.1 Ausencia de data leakage

La separación temporal garantiza que ningún dato generado después de la cirugía (que pudiera revelar el desenlace) contamina los datos de entrada del modelo predictivo. La integridad se ha verificado fila a fila sobre 4,294,675 registros temporales.

### 6.2 Representatividad del pre-operatorio

El segmento pre-operatorio cubre al 100% de los 12,288 pacientes con cirugía de referencia. Esto significa que para todos los pacientes existe al menos un registro clínico anterior a la cirugía, condición necesaria para poder construir características predictoras.

### 6.3 Limitaciones asumidas

1. **Resolución temporal de minuto**: los timestamps del hospital tienen resolución de minuto. Los eventos registrados exactamente en `t_inicio` o `t_fin` se asignan a intra-operatorio (criterio de intervalo cerrado $[t_i, t_f]$), lo que es clínicamente consistente.
2. **Unicidad del episodio de referencia**: se usa la primera cirugía mayor/principal de cada paciente. Pacientes con múltiples cirugías en el historial contribuyen con los datos de todas sus intervenciones al conjunto (como pre/post relativo a la cirugía elegida), lo que podría introducir confusión en el seguimiento longitudinal. Este aspecto queda fuera del alcance del TFG y se documenta como limitación.
3. **213 pacientes sin post-op**: ausencia de datos posteriores a la cirugía para el 1.73% de pacientes. No afecta al pre-operatorio (base del modelo predictivo) pero limita el análisis de evolución post-quirúrgica para ese subgrupo.

---

## 7. Conclusión

El pipeline de separación temporal ha sido diseñado, auditado y verificado satisfactoriamente. Los 4 bugs detectados y corregidos durante la auditoría habrían supuesto:

- La pérdida de información diagnóstica relevante (bugs 1 y 2).
- La ausencia total de los registros intra-operatorios (bug 3) — el error más grave, con 12,288 filas perdidas silenciosamente.
- Data leakage por contaminación cruzada pre/post en 2 pacientes (bug 4).

Con el pipeline corregido, el dataset producido satisface los requisitos de **completitud** (46/46 archivos, 5,144,115 filas), **integridad temporal** (0 violaciones en 4.3M filas), **mutua exclusividad** (sin filas en dos segmentos simultáneamente) y **trazabilidad** (toda anomalía rastreada a su origen en los datos hospitalarios).

---

## Apéndice A — Inventario de Archivos por Segmento

### static/ (archivos sin dimensión temporal)
| Archivo | Descripción |
|---|---|
| `000_POBLACION_DIANA.csv` | Características demográficas y baseline |
| `001_INTERVENCIONES_QUIRURGICAS.csv` | Codificación de la intervención |
| `004_DIAGNOSTICO_PRINCIPAL_CIE10.csv` | Diagnóstico principal CIE-10 |
| `005_DIAGNOSTICO_PRINCIPAL_CIE9.csv` | Diagnóstico principal CIE-9 |
| `006_COMORBILIDADES_CHARLSON.csv` | Índice de Charlson |
| `006_COMORBILIDADES_CHARLSON_NUEVO.csv` | Índice de Charlson (actualización) |
| `021_DIAGNOSTICOS_POR_PACIENTE.csv` | Historial diagnóstico completo |
| `028_INTERVENCIONES_CHECKLIST_CIRUGIA_MAYOR.csv` | Checklist cirugía mayor |
| `029_INTERVENCIONES_CHECKLIST_CIRUGIA_MENOR.csv` | Checklist cirugía menor |

### Criterios de asignación de archivos dinámicos

Los 37 archivos temporales se asignan a los segmentos según la fecha de cada evento respecto a la ventana quirúrgica del paciente. Un archivo puede aparecer en múltiples segmentos (pre, intra, post y/o sin_referencia) si contiene registros del mismo paciente en distintos momentos.

---

## Apéndice B — Reproducibilidad

El pipeline completo es reproducible ejecutando las celdas del notebook `notebooks/ntb_01_separacion_datos.ipynb` en secuencia. Los parámetros clave son:

| Parámetro | Valor |
|---|---|
| Ruta datos raw | `data/data_raw/` |
| Ruta salida normalized | `data/data_normalized/` |
| Ruta salida dividido | `data/data_divided/` |
| Archivo de referencia | `009_INTERVENCIONES_QUIRURGICAS_INTERVENCION.csv` |
| Criterio episodio | Primera CIRUGIA MAYOR PRINCIPAL; si no, primera Principal |
| Criterio pre | `fecha_evento < Hora (inicio)` |
| Criterio intra | `Hora (inicio) ≤ fecha_evento ≤ Hora Intervención (fin)` |
| Criterio post | `fecha_evento > Hora Intervención (fin)` |
| Entorno Python | `tfg_ml` (conda), Python 3.11.15 |
