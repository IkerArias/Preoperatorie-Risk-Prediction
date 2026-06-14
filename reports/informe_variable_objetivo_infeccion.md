# Informe: Construcción de la Variable Objetivo de Infección Postoperatoria

**TFG — Predicción de Complicaciones Postquirúrgicas**  
**Notebook**: `ntb_04_variable_objetivo_infeccion.ipynb`  
**Fecha**: Abril 2026

---

## 1. Objetivo

El objetivo de este notebook es construir la **variable binaria de infección postoperatoria** (`infeccion_postop`) que servirá como variable dependiente (`y`) en los modelos de clasificación.

El planteamiento es el siguiente: dado un paciente intervenido quirúrgicamente, ¿desarrolló una infección documentada tras la intervención?

- `y = 1` → el paciente tuvo al menos un diagnóstico CIE-10 de infección registrado en el alta hospitalaria
- `y = 0` → no se registró ningún diagnóstico de infección

---

## 2. Fuentes de datos utilizadas

### 2.1. `data/data_divided/static/000_POBLACION_DIANA.csv`

| Atributo | Valor |
|---|---|
| **Separador** | Punto y coma (`;`) |
| **Filas** | 13,662 |
| **Columnas** | `Id_Paciente`, `sexo`, `fecnac`, `fecdef`, `residen`, `escalon` |
| **Uso en el notebook** | Define la **población de referencia**: la lista completa de pacientes del estudio. Cada paciente de esta lista recibirá exactamente una fila en la variable objetivo, con `y=0` por defecto. |

Este archivo es fundamental porque **marca el denominador del estudio**: cualquier paciente presente aquí que no aparezca en diagnósticos de infección se considera negativo (no ocurrió infección).

---

### 2.2. `data/data_divided/post/016_HOSPITALIZACION_DIAGNOSTICO_PRINCIPAL.csv`

| Atributo | Valor |
|---|---|
| **Separador** | Coma |
| **Filas** | 4,024 |
| **Pacientes únicos** | 2,800 |
| **Rango temporal** | 2019-01-11 — 2020-07-17 |
| **Columnas** | `id_paciente`, `id_episodio`, `fecha_evento`, `cod Diagnóstico CIE10`, `Diagnóstico CIE10`, `Órden diagnóstico`, `Fecha (alta)`, `Servicio HOSP (ingreso)`, `Circunstancia alta` |
| **Uso en el notebook** | Fuente principal para identificar infecciones. Cada fila es un diagnóstico CIE-10 registrado al alta de un episodio de hospitalización. Un mismo paciente puede tener múltiples filas (varios episodios, varios diagnósticos por episodio). |

**¿Por qué este archivo?**  
Los diagnósticos al alta hospitalaria (`post/016_*`) son el registro clínico oficial de las complicaciones ocurridas durante o tras la hospitalización quirúrgica. El código CIE-10 es el estándar internacional de clasificación de enfermedades empleado en los sistemas hospitalarios de salud.

> **Nota sobre la segmentación temporal**: este archivo está en la carpeta `post/` porque se trata de información generada *después* de la intervención quirúrgica (diagnósticos al alta). Incluirlo como feature predictiva sería **data leakage**; por eso se usa exclusivamente para construir la variable objetivo `y`, nunca como predictor.

---

## 3. Metodología de clasificación CIE-10

### 3.1. ¿Qué es un código CIE-10?

La Clasificación Internacional de Enfermedades (CIE-10 / ICD-10) es un sistema jerárquico donde cada código identifica una enfermedad o condición clínica específica. Los primeros caracteres del código definen la categoría general:

- `A40`, `A41` → enfermedades infecciosas sistémicas (sepsis)
- `J10`–`J49` → enfermedades del aparato respiratorio (infecciones pulmonares)
- `T81` → complicaciones de procedimientos (infecciones de herida quirúrgica)
- `N39` → enfermedades urinarias (infección de tracto urinario)

### 3.2. Taxonomía de categorías de infección postoperatoria

Se definieron **6 categorías clínicas** mediante expresiones regulares aplicadas sobre el código CIE-10 normalizado (mayúsculas, sin espacios):

| Prioridad | Categoría | Descripción clínica | Regex CIE-10 | Ejemplos reales encontrados |
|---|---|---|---|---|
| 1 | `cateter` | Infección por catéter/dispositivo | `^(T802\|T827\|T857\|Z5111)` | Z5111 (63 casos), T827XXA (6) |
| 2 | `sepsis` | Sepsis o septicemia | `^(A40\|A41)` | A419 (54), A4151 (11), A403 (5) |
| 3 | `ssi` | Infección de sitio quirúrgico | `^(T814\|T818\|T819)` | T814XXA (36), T8189XA (5) |
| 4 | `itu` | Infección tracto urinario | `^N390` | N390 (56 casos) |
| 5 | `resp` | Infección respiratoria | `^(J1[0-9]\|J2[0-9]\|J4[0-9])` | J441 (39), J189 (37), J22 (32) |
| 6 | `other` | Otras infecciones clínicas relevantes | `^(L03\|K61\|K941\|J36\|N41\|K65\|M00\|M86\|G00\|G06)` | L03115 (5), K610 (4) |

**Justificación del orden de prioridad**: si un paciente tiene múltiples diagnósticos de infección (por ejemplo, catéter Y sepsis), se le asigna la categoría clínicamente más específica/grave. El cateterismo precede porque es etiología directa del proceso quirúrgico; la sepsis es la más grave de las infecciones sistémicas.

### 3.3. Proceso de clasificación paso a paso

```
Para cada fila de 016_DIAGNOSTICO_PRINCIPAL:
  1. Normalizar código: mayúsculas, strip espacios → "T814XXA"
  2. Aplicar regex en orden de prioridad (cateter → sepsis → ssi → itu → resp → other)
  3. Asignar la primera categoría que coincida (None si ninguna coincide)

Para cada paciente (agrupando):
  4. Si tiene ≥1 diagnóstico de infección → tomar la categoría de MAYOR prioridad (menor índice)
  5. Si no tiene ninguno → "ninguna"
```

**Left join sobre la población diana**: garantiza que los 13,662 pacientes del estudio tengan exactamente una fila en `y`, con `y=0` para los que no aparecen en diagnósticos de infección.

---

## 4. Resultados

### 4.1. Estadísticas de la variable objetivo

| Métrica | Valor |
|---|---|
| **Total pacientes** | 13,662 |
| **Positivos (y=1)** | 413 (3.02%) |
| **Negativos (y=0)** | 13,249 (96.98%) |
| **Ratio neg/pos** | ~32.1 : 1 |

### 4.2. Distribución por categoría de infección

| Categoría | Nº pacientes | % sobre positivos | Descripción |
|---|---|---|---|
| `resp` | 193 | 46.7% | Neumonía, bronquitis aguda, EPOC infectado |
| `sepsis` | 67 | 16.2% | Sepsis / septicemia bacteriana |
| `itu` | 53 | 12.8% | Infección de tracto urinario |
| `ssi` | 40 | 9.7% | Infección de herida / sitio quirúrgico |
| `cateter` | 35 | 8.5% | Infección por catéter o dispositivo |
| `other` | 25 | 6.1% | Otras (celulitis, absceso perineal, retención de absceso, etc.) |

### 4.3. Códigos CIE-10 más frecuentes por categoría

**RESP (193 pacientes)**:
- `J441` — Enfermedad pulmonar obstructiva crónica con infección aguda (39)
- `J189` — Neumonía no especificada (37)
- `J22` — Infección aguda no especificada de vías respiratorias inferiores (32)
- `J1289` — Gripe con otras manifestaciones respiratorias (28)

**SEPSIS (67 pacientes)**:
- `A419` — Septicemia no especificada (54)
- `A4151` — Sepsis por Pseudomonas (11)
- `A403` — Sepsis por Streptococcus pneumoniae (5)

**ITU (53 pacientes)**:
- `N390` — Infección de vías urinarias, sitio no especificado (56)

**SSI (40 pacientes)**:
- `T814XXA` — Infección consecutiva a procedimiento, encuentro inicial (36)
- `T8189XA` — Otras complicaciones de procedimientos (5)

**CATÉTER (35 pacientes)**:
- `Z5111` — Encuentro para quimioterapia antineoplásica (63) — indica hospitalización por complicación de dispositivo
- `T827XXA` — Infección/reacción inflamatoria por prótesis cardiaca (6)

**OTHER (25 pacientes)**:
- `L03115` — Celulitis de extremidad inferior derecha (5)
- `K610` — Absceso del esfínter anal (4)

---

## 5. Desbalanceo de clases y su impacto en el modelado

El ratio 32:1 es el principal reto técnico del problema:

```
Negativos: ████████████████████████████████  96.98% (13,249)
Positivos: █                                   3.02%    (413)
```

Este nivel de desbalanceo implica que:

1. Un clasificador trivial que prediga siempre "no infección" obtendría 96.98% de accuracy
2. Las métricas de accuracy son **engañosas** → usar PR-AUC y Recall como métricas principales
3. Es **obligatorio** aplicar técnicas de balanceo (SMOTE, ADASYN, class_weight) durante el entrenamiento

**Técnicas de balanceo previstas** en el notebook `ntb_04_construccion_features_infeccion.ipynb`:
- SMOTE (Synthetic Minority Over-sampling Technique)
- GradientBoosting con `scale_pos_weight`
- RandomForest con `class_weight="balanced"`

---

## 6. Archivo de salida

| Archivo | Ruta | Columnas | Filas |
|---|---|---|---|
| `y_infeccion_postop.csv` | `results/y_infeccion_postop.csv` | `id_paciente`, `infeccion_postop`, `categoria_infeccion` | 13,662 |

Este CSV es el **input directo** del notebook `ntb_04_construccion_features_infeccion.ipynb`, donde se construye la matriz de features `X` y se entrenan los modelos predictivos.

---

## 7. Limitaciones y consideraciones

1. **Cobertura parcial de 016**: el archivo tiene 4,024 filas con 2,800 pacientes únicos, pero la población diana es de 13,662. Esto significa que 10,862 pacientes (79.5%) no tienen ningún diagnóstico al alta en este archivo — se consideran negativos, lo que es correcto desde la perspectiva de predicción de complicaciones.

2. **Sesgo de registro**: algunos diagnósticos de infección podrían no haberse codificado en CIE-10 al alta, resultando en falsos negativos. Este sesgo es inherente a los datos clínicos administrativos.

3. **Temporalidad**: el archivo `post/016_*` solo contiene episodios de hospitalización *después* de la cirugía. No se incluyen infecciones diagnosticadas en atención primaria o ambulatoria.

4. **Categoría `resp` dominante**: la alta frecuencia de infecciones respiratorias (47% de positivos) puede estar parcialmente influida por pacientes con EPOC preexistente (`J441`) que presentan exacerbaciones infecciosas, no necesariamente relacionadas con la cirugía. Esto es una limitación a mencionar en el análisis de resultados del TFG.
