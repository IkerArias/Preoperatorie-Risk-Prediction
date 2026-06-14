# Informe: Construcción de la Matriz de Features — Infección Postoperatoria

**TFG — Predicción de Complicaciones Postquirúrgicas**  
**Notebook**: `ntb_04_construccion_features_infeccion.ipynb`  
**Fecha**: Abril 2026

---

## 1. Objetivo

El objetivo de este notebook es construir la **matriz de features X** (una fila por paciente, una columna por variable predictora) que, combinada con la variable objetivo `y` construida en `ntb_03b`, forma el dataset completo para entrenar los modelos de clasificación.

La pregunta que responde este notebook es:
> *¿Qué información clínica disponible ANTES o DURANTE la cirugía puede predecir si un paciente desarrollará una infección postoperatoria?*

La restricción fundamental es la **separación temporal**: solo podemos usar información del período preoperatorio (`pre/`) e intraoperatorio (`intra/`) — nunca datos postoperatorios, que serían data leakage.

---

## 2. Resultado final del dataset

| Atributo | Valor |
|---|---|
| **Pacientes** | 13,662 |
| **Features (columnas predictoras)** | 30 |
| **Positivos (infección = 1)** | 413 (3.02%) |
| **Negativos (no infección = 0)** | 13,249 (96.98%) |
| **Ratio desbalanceo** | ~32 : 1 |
| **Valores nulos tras imputación** | 0 |
| **Archivos exportados** | `results/X_infeccion.csv`, `results/y_infeccion_postop.csv` |

---

## 3. Fuentes de datos y features construidas

### 3.1. Features demográficas — `000_POBLACION_DIANA.csv` (static/)

**2 features**: `edad`, `sexo_mujer`

La **edad** no se extrae directamente del CSV porque el archivo solo contiene `fecnac` (fecha de nacimiento). Para calcular la edad en el momento de riesgo real, se realiza un merge con `009_INTERVENCIONES_QUIRURGICAS_INTERVENCION.csv` para obtener la `fecha_cirugia`:

```
edad = (fecha_cirugia − fecnac) / 365.25
```

Esto es importante: un paciente que se operó en 2019 con 60 años no debería recibir la edad actual (2026) como feature — introduciría sesgo temporal.

El **sexo** se codifica como variable binaria: `sexo_mujer = 1` si mujer, `0` si hombre.

---

### 3.2. Comorbilidades de Charlson — `006_COMORBILIDADES_CHARLSON_NUEVO.csv` (static/)

**15 features**: `charlson_score` + 14 condiciones específicas

El **Índice de Charlson** es el estándar clínico internacional para cuantificar el peso de las enfermedades crónicas de un paciente y predecir mortalidad a 1 año. A mayor puntuación, mayor morbilidad asociada.

Las 14 condiciones adicionales son enfermedades frecuentes en la población quirúrgica con impacto demostrado en la recuperación postoperatoria: HTA (91% de los pacientes), dislipemia (89%), arritmia, asma, infección bronquial crónica, bronquiectasia, coagulopatía, etc.

**Preprocesado**: el archivo puede contener múltiples registros por paciente (actualizaciones anuales). Se toma el `.max()` por `id_paciente` para conservar la comorbilidad más grave registrada en cualquier momento.

---

### 3.3. Features quirúrgicas — `009_INTERVENCIONES_QUIRURGICAS_INTERVENCION.csv` (intra/)

**7 features**: `duracion_min`, `asa`, `urgente`, `cirugia_mayor`, `infeccion_intraop`, `anestesia_general`, `anestesia_locoregional`

| Feature | Tipo | Descripción |
|---|---|---|
| `duracion_min` | Numérica | Duración de la cirugía en minutos |
| `asa` | Ordinal (1–6) | Riesgo anestésico ASA. Evalúa el estado físico global del paciente antes de anestesia |
| `urgente` | Binaria | 1 si la cirugía fue urgente/emergente (menos preparación preoperatoria) |
| `cirugia_mayor` | Binaria | 1 si es cirugía mayor (mayor trauma tisular, mayor riesgo infeccioso) |
| `infeccion_intraop` | Binaria | 1 si se documentó contaminación intraoperatoria |
| `anestesia_general` | Binaria | 1 si se usó anestesia general (mayor impacto inmunosupresor) |
| `anestesia_locoregional` | Binaria | 1 si fue intradural, epidural, bloqueo de plexo o local |

El **ASA** (American Society of Anesthesiologists Physical Status) es uno de los predictores más robustos de complicaciones postquirúrgicas en la literatura clínica.

---

### 3.4. Laboratorio preoperatorio — `010_LABORATORIO*.csv` (pre/)

**0 features en la versión final** (por insuficiente cobertura)

Se concatenaron todos los archivos `010_LABORATORIO*.csv` del directorio `pre/` y se filtraron 13 analíticas de relevancia clínica para infección:
- Hemograma: Hemoglobina, Leucocitos, Neutrófilos, Linfocitos, Hematocrito, Plaquetas
- Bioquímica: Creatinina, Glucosa, Sodio, Potasio, Urea
- Inflamación: Proteína C Reactiva (PCR)
- Coagulación: Índice de protrombina

**Resultado**: solo **58 pacientes** (0.4% de 13,662) tenían alguna de estas analíticas registradas en los archivos `pre/`. Al aplicar el criterio de eliminar features con >80% de valores nulos, **todas las columnas de laboratorio fueron descartadas**.

> **Interpretación**: los archivos `pre/010_LABORATORIO*.csv` son una muestra muy pequeña del total de analíticas preoperatorias. La mayoría de las analíticas pre-cirugía estarán en otros archivos del sistema o no están segmentadas en la carpeta `pre/`. Esta es la **principal limitación del dataset** desde el punto de vista clínico, ya que la PCR y el hemograma son los marcadores más directos de riesgo infeccioso.

---

### 3.5. Constantes vitales preoperatorias — `013_CONSTANTES*.csv` (pre/)

**3 features en la versión final** (de 9 planificadas)

| Item planificado | Cobertura real | En X final |
|---|---|---|
| Presión arterial sistólica | 84.1% (11,483 pac.) | ✅ `vital_ta_sist` |
| Presión arterial diastólica | 84.0% (11,482 pac.) | ✅ `vital_ta_dias` |
| Frecuencia cardiaca | 83.4% (11,390 pac.) | ✅ `vital_fc` |
| Temperatura | 0% | ❌ descartada |
| Saturación O2 | 0% | ❌ descartada |
| Glucemia basal | 0% | ❌ descartada |
| Peso (kg) | 0% | ❌ descartada |
| Talla (cm) | 0% | ❌ descartada |
| IMC | 0% | ❌ descartada |

Los ítems con 0% de cobertura probablemente tienen nombres ligeramente distintos en los archivos reales (diferencias en mayúsculas, tildes o formato). TA y FC sí coincidieron exactamente y tienen cobertura del ~84%.

Los datos de constantes están en **formato largo** (tall format): cada medición es una fila. El notebook:
1. Filtra los ítems de interés
2. Toma el **último valor registrado** antes de la cirugía (`last()` tras ordenar por fecha)
3. Pivota a formato ancho (`unstack`)

---

### 3.6. Solicitud quirúrgica — `007_INTERVENCIONES_QUIRURGICAS_SOLICITUD.csv` (pre/)

**3 features**: `asa_solicitud`, `ingreso_urgente`, `suspension_farmacos`

La solicitud quirúrgica se cumplimenta **antes de la intervención** (es la petición del cirujano). Complementa los datos del acto quirúrgico:
- `asa_solicitud`: valoración ASA del cirujano solicitante (puede diferir del anestesista)  
- `ingreso_urgente`: si el tipo de ingreso fue urgente
- `suspension_farmacos`: si se indicó suspensión de anticoagulantes/antiagregantes preoperatoria

---

### 3.7. Merge final y gestión de missings

Todos los bloques se unen mediante **left join sobre `id_paciente`**, anclado en la lista de 13,662 pacientes de la variable objetivo. Esto garantiza que todos los pacientes tienen una fila, aunque no tengan datos en alguna fuente (quedan como `NaN`).

**Imputación**: se aplica imputación por **mediana** (robusta frente a valores extremos clínicos) con `sklearn.SimpleImputer`. Ninguna feature supera el 80% de nulos, por lo que no se descarta ninguna columna en esta versión.

---

## 4. Análisis descriptivo del dataset final

### 4.1. Resumen de las 30 features

| Grupo | Features | Nº |
|---|---|---|
| Demográficas | `edad`, `sexo_mujer` | 2 |
| Charlson | `charlson_score`, `EII`, `angina`, `arritmia`, `hta`, `dislipemia`, `linfoma`, `leucemia`, `coagulopatia`, `sangrado_gastro`, `asma`, `bronquiectasia`, `fibrosis_quis`, `enf_pulmon_intersticial`, `inf_bronquial_cronica` | 15 |
| Quirúrgicas | `duracion_min`, `asa`, `urgente`, `cirugia_mayor`, `infeccion_intraop`, `anestesia_general`, `anestesia_locoregional` | 7 |
| Constantes vitales | `vital_ta_sist`, `vital_ta_dias`, `vital_fc` | 3 |
| Solicitud | `asa_solicitud`, `ingreso_urgente`, `suspension_farmacos` | 3 |

---

### 4.2. Comparativa de features entre infectados y no infectados

#### Variables continuas (test t de Student)

| Feature | Infectados (y=1) | No infectados (y=0) | p-valor | Significación |
|---|---|---|---|---|
| **Edad** | 71.5 ± 13.8 años | 61.9 ± 16.2 años | < 0.001 | *** |
| **Charlson score** | 2.69 ± 1.83 | 2.02 ± 1.10 | < 0.001 | *** |
| **Duración cirugía** | 93.9 ± 87.1 min | 74.1 ± 65.3 min | < 0.001 | *** |
| **ASA** | 2.60 ± 0.94 | 1.98 ± 0.78 | < 0.001 | *** |
| **Frecuencia cardiaca** | 74.5 lpm | 72.9 lpm | 0.012 | * |
| **TA sistólica** | 142.8 mmHg | 141.0 mmHg | 0.153 | ns |
| **TA diastólica** | 77.4 mmHg | 80.9 mmHg | 0.293 | ns |

Las 4 variables más discriminativas son estadísticamente significativas con p < 0.001: **edad, charlson_score, duración y ASA**. Son exactamente los factores de riesgo descritos en la literatura de complicaciones postquirúrgicas.

#### Variables binarias notables

| Feature | Infectados (y=1) | No infectados (y=0) | Diferencia |
|---|---|---|---|
| Sexo mujer | 34.1% | 48.4% | -14.3 pp |
| Urgente | **12.3%** | 6.8% | +5.5 pp |
| Anestesia general | 27.1% | 22.3% | +4.8 pp |
| Cirugía mayor | 75.3% | 78.6% | -3.3 pp |
| Suspensión fármacos | **27.1%** | 15.1% | +12.0 pp |
| Inf. bronquial crónica | **8.2%** | 2.5% | +5.8 pp |
| Asma | 8.2% | 3.6% | +4.6 pp |
| Angina | 5.1% | 1.3% | +3.7 pp |

**Hallazgo relevante**: los pacientes infectados tienen una menor proporción de mujeres (34% vs 48%). Esto es consistente con la literatura: las mujeres tienen menor incidencia de infecciones de herida en cirugía (menor masa muscular, diferencias hormonales en respuesta inmune).

**Suspensión de fármacos** (anticoagulantes/antiagregantes): casi el doble en infectados (27.1% vs 15.1%), lo que puede reflejar que son pacientes con mayor comorbilidad cardiovascular.

---

### 4.3. Distribución del Charlson Score

El Charlson Score de los infectados está sesgado hacia valores más altos. Aunque el valor modal es 2 en ambos grupos (porque la mayoría de pacientes tienen HTA + dislipemia que contribuyen al score), los infectados tienen mayor presencia en scores ≥ 4:

| Score | Infectados | No infectados |
|---|---|---|
| 0 | 2.2% | 7.2% |
| 1 | 2.4% | 4.8% |
| **2** | **74.8%** | **79.9%** |
| ≥ 3 | 20.6% | 8.1% |

---

## 5. Limitaciones identificadas

### 5.1. Ausencia de laboratorio preoperatorio (limitación principal)

Solo el **0.4%** de los pacientes tiene analíticas registradas en `pre/010_LABORATORIO*.csv`. Esto es clínicamente relevante porque:
- La PCR (Proteína C Reactiva) es el marcador inflamatorio más utilizado en clínica para estratificar riesgo infeccioso
- El recuento de leucocitos y neutrófilos son los indicadores clásicos de infección/inflamación
- La glucemia preoperatoria es un factor de riesgo de SSI bien documentado (hiperglucemia + cirugía = riesgo aumentado)

**Posibles causas**: los archivos `pre/010_LABORATORIO*.csv` pueden ser una fracción de los datos reales (ej: solo un año o un servicio específico). Las analíticas preoperatorias completas podrían estar en otros archivos del sistema de información hospitalario.

### 5.2. Constantes vitales incompletas

Temperatura, SatO2, IMC y glucemia basal tienen 0% de cobertura en los archivos de constantes. Probablemente los nombres de los ítems en el sistema real difieren ligeramente de los usados en el filtrado. En un proyecto real habría que validar los nombres exactos con el equipo clínico.

### 5.3. Infección intraoperatoria sin variabilidad

La variable `infeccion_intraop` (infección documentada durante la cirugía) presenta un 0% de positivos en ambos grupos, lo que indica que este campo no se registra sistemáticamente en el sistema de información hospitalario. No aportará información predictiva al modelo.

### 5.4. Desbalanceo 32:1

Con solo el 3.02% de casos positivos, el dataset está **fuertemente desbalanceado**. Cualquier modelo entrenado sin gestión del desbalanceo predirá casi siempre "no infección" y obtendrá ~97% de accuracy de forma trivial. La sección 10 del notebook aborda esto con SMOTE.

---

## 6. Conclusión

El dataset final resulta en una matriz de **13,662 pacientes × 30 features** con **0 valores nulos** (tras imputación por mediana). Las features más prometedoras para el modelo, según el análisis descriptivo, son:

1. **Edad** (mayor en infectados: +9.6 años, p < 0.001)
2. **ASA** (escala de riesgo anestésico, +0.62 puntos en infectados, p < 0.001)
3. **Charlson score** (mayor carga de comorbilidad, +0.67 puntos, p < 0.001)
4. **Duración de la cirugía** (+19.8 minutos en infectados, p < 0.001)
5. **Infección bronquial crónica** (+5.8 pp en infectados)
6. **Cirugía urgente** (+5.5 pp en infectados)

La principal limitación es la ausencia de analíticas de laboratorio preoperatorias (PCR, leucocitos, glucosa), que en la práctica clínica son los predictores más usados para estratificar el riesgo infeccioso.

---

## 7. Archivos generados

| Archivo | Ruta | Descripción |
|---|---|---|
| `X_infeccion.csv` | `results/X_infeccion.csv` | Matriz de features: 13,662 × 31 (incluye id_paciente) |
| `y_infeccion_postop.csv` | `results/y_infeccion_postop.csv` | Variable objetivo: 13,662 × 3 (id, y, categoría) |

Estos archivos son la **entrada directa de la sección 10** del notebook, donde se entrenan y evalúan los modelos de clasificación.
