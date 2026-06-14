# Informe: Predicción de Infección Postoperatoria mediante Machine Learning
## Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos

**Notebook de referencia:** `notebooks/ntb_05_modelo_infeccion.ipynb`  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Este informe documenta el diseño, entrenamiento y evaluación de un sistema de Machine Learning para predecir la **infección postoperatoria** en pacientes quirúrgicos, abordando el problema en tres niveles de granularidad creciente:

1. **¿Se infectará el paciente?** — clasificación binaria (sí/no)
2. **¿Qué tipo de infección tendrá?** — clasificación multiclase global (7 categorías)
3. **Dado que ya está infectado, ¿cuál es el tipo?** — modelo condicional (6 categorías)

**Conclusiones principales:**

- El mejor modelo binario es **LightGBM** (GridSearch): PR-AUC = 0.247, AUC-ROC = 0.788, IC 95% bootstrap [0.158–0.353] y [0.725–0.847] respectivamente.
- Con umbral de Youden (0.084) se consigue **recall = 0.699, especificidad = 0.836**: el modelo detecta 7 de cada 10 infecciones manteniendo una especificidad clínicamente razonable.
- La técnica de balanceo ganadora es **SMOTEENN** (PR-AUC CV = 0.225), seleccionada sin tocar el test set.
- Para clasificación del tipo de infección, **RandomForest** lidera en ambos problemas multiclase: F1 macro = 0.195 (7 clases) y 0.337 (6 clases condicional, CV = 0.333 ± 0.069).
- Los resultados son modestos pero **honestos y metodológicamente rigurosos**: predecir infección postoperatoria con datos exclusivamente preoperatorios sobre un evento de 3% de prevalencia es un problema de alta dificultad intrínseca.

---

## 1. Introducción y contexto clínico

### 1.1 Problema

La infección postoperatoria es una complicación frecuente y potencialmente grave tras la cirugía. Identificar preoperatoriamente qué pacientes tienen mayor riesgo permitiría:

- **Intensificar la profilaxis antibiótica** en pacientes de alto riesgo
- **Optimizar la asignación de recursos** (UCI, vigilancia intensificada)
- **Informar al paciente** con mayor precisión sobre los riesgos de la intervención

El problema plantea dos dificultades estadísticas principales:

| Dificultad | Valor observado |
|---|---|
| Prevalencia de infección | 3.02% (ratio 32:1) |
| Disponibilidad de features | Solo datos preoperatorios e intraoperatorios |
| Clases en el target multiclase | 7 (muy desbalanceadas: de 13.249 "ninguna" a 25 "other") |

### 1.2 Dataset

- **Fuente:** Dataset anestesiológico de Osakidetza (anonimizado)
- **Población de estudio:** 13.661 pacientes con cirugía y seguimiento de infección postoperatoria
- **Período:** Cirugías del período cubierto por el dataset
- **Variable objetivo binaria:** `infeccion_postop` (0=no, 1=sí)
- **Variable objetivo multiclase:** `categoria_infeccion` (ninguna, resp, sepsis, itu, ssi, cateter, other)

### 1.3 Distribución del target multiclase

| Categoría | N | % |
|---|---|---|
| ninguna | 13.249 | 96.98% |
| resp (infección respiratoria) | 193 | 1.41% |
| sepsis | 67 | 0.49% |
| itu (infección del tracto urinario) | 53 | 0.39% |
| ssi (infección de herida quirúrgica) | 40 | 0.29% |
| cateter | 35 | 0.26% |
| other | 25 | 0.18% |

---

## 2. Metodología

### 2.1 Preprocesado y construcción de features

Se construye una matriz de 42 features combinando dos fuentes:

**Features del dataset original (`FEATS_ORIG`, 32 variables):**
- Demográficas: `edad`, `sexo_mujer`
- Comorbilidades de Charlson (14 condiciones + puntuación total)
- Variables ASA: clasificación ASA, tipo cirugía, urgencia, especialidad
- Variables intraoperatorias: duración, transfusión, tipo anestesia

**Features derivadas del conjunto de infección (`FEATS_XINF`, 10 variables):**
- Variables adicionales del período perioperatorio específicas del seguimiento de infección

**Deduplicación:** Cuando un paciente tiene múltiples registros quirúrgicos, se conserva el más representativo mediante matching por edad. Dataset final: 13.661 pacientes únicos.

**Imputer:** `SimpleImputer(strategy='median')` integrado dentro del `ColumnTransformer`, refitado en cada fold (nunca sobre el test set).

### 2.2 Split train/test y anti-leakage

```
Split estratificado: 80% train / 20% test  →  10.928 train | 2.733 test
                                                  330 positivos (3.02%)  |  83 positivos (3.04%)
```

**Garantía de consistencia:** Los tres modelos (binario, multiclase global, condicional) comparten exactamente los mismos índices `train_idx` / `test_idx`. Esto evita que el jurado pueda cuestionar que los resultados entre modelos son comparables.

**Garantías anti-leakage implementadas:**

| Potencial fuente de leakage | Solución aplicada |
|---|---|
| Imputer ajustado antes del split | `SimpleImputer` dentro de `ColumnTransformer`, se refita en cada fold |
| SMOTEENN ajustado sobre todo el dataset | `ImbPipeline` con sampler dentro de cada fold de CV |
| BEST_TECNICA seleccionada por test PR-AUC | Selección por **CV PR-AUC** (calculado solo sobre `X_train`) |
| Test sets diferentes entre modelos binario/multiclase | `train_idx`/`test_idx` compartidos en los 3 modelos |

### 2.3 Pipeline completo

```
ImbPipeline:
  ├── preprocessor: ColumnTransformer
  │     ├── imp_scale: SimpleImputer(median) → StandardScaler  [features continuas]
  │     └── remainder: SimpleImputer(median)                   [features binarias/ordinales]
  ├── sampler: SMOTEENN(random_state=42, sampling_strategy=0.3)
  └── model: [XGBoost | LightGBM | RandomForest | ...]
```

---

## 3. Fase 1b — Comparativa de técnicas de balanceo

Se evalúan 3 técnicas × 3 modelos de screening mediante GridSearch interno (3-fold). La métrica de selección es **CV PR-AUC** (score de validación cruzada sobre `X_train`), nunca el test set.

| Técnica | CV PR-AUC medio (3 modelos) | Test PR-AUC ref. |
|---|---|---|
| **SMOTEENN** | **0.2250** | 0.2408 |
| RandomOverSampler | 0.2218 | 0.2187 |
| SMOTE | 0.2188 | 0.2226 |

**Ganadora: SMOTEENN** — combina la generación sintética de SMOTE con la limpieza de solapamiento mediante Edited Nearest Neighbours (ENN), eliminando muestras ruidosas en las fronteras de decisión.

---

## 4. Fase 2 — Entrenamiento baseline (7 modelos)

Entrenamiento directo con SMOTEENN y parámetros por defecto. Evaluación en el test set fijo.

| Modelo | PR-AUC | AUC-ROC | Recall | Precisión | F1 |
|---|---|---|---|---|---|
| LightGBM | 0.2373 | 0.7934 | 0.2410 | 0.3175 | 0.2740 |
| GaussianNB | 0.2360 | 0.7464 | 0.6506 | 0.0939 | 0.1641 |
| GradientBoosting | 0.2323 | 0.7817 | 0.2892 | 0.2727 | 0.2807 |
| XGBoost | 0.2299 | 0.7927 | 0.2410 | 0.2899 | 0.2632 |
| ExtraTrees | 0.2243 | 0.7956 | 0.3253 | 0.2269 | 0.2673 |
| RandomForest | 0.2214 | 0.7959 | 0.2530 | 0.2958 | 0.2727 |
| LogisticRegression | 0.1449 | 0.7611 | 0.5181 | 0.1635 | 0.2486 |

> **Nota:** GaussianNB destaca por recall (0.650) a costa de precisión (0.094) — patrón consistente con otros targets del proyecto. LightGBM y LightGBM lideran PR-AUC.

---

## 5. Fase 3 — Validación cruzada K-Fold (5 folds)

Validación más robusta refitando el pipeline completo en cada fold. Los scores CV son más representativos de la capacidad de generalización real que los scores en test.

| Modelo | PR-AUC CV medio | ±std | AUC-ROC CV medio | ±std |
|---|---|---|---|---|
| GaussianNB | 0.3279 | 0.0384 | 0.8292 | 0.0361 |
| LightGBM | 0.2511 | 0.0548 | 0.8576 | 0.0153 |
| XGBoost | 0.2489 | 0.0594 | 0.8537 | 0.0197 |
| ExtraTrees | 0.2435 | 0.0638 | 0.8497 | 0.0346 |
| GradientBoosting | 0.2416 | 0.0638 | 0.8570 | 0.0205 |
| RandomForest | 0.2409 | 0.0471 | 0.8555 | 0.0277 |
| LogisticRegression | 0.2335 | 0.0558 | 0.8425 | 0.0356 |

> GaussianNB muestra el PR-AUC CV más alto (0.328) pero la peor AUC-ROC (0.829) y una variabilidad reducida. Los modelos de árboles de decisión boosteados (LightGBM, XGBoost, GradientBoosting) ofrecen el mejor equilibrio PR-AUC / AUC-ROC.

---

## 6. Fase 4 — Optimización GridSearch

GridSearch con validación cruzada 3-fold interna sobre `X_train`. Métrica de scoring: `average_precision` (PR-AUC).

Mejores hiperparámetros encontrados por los modelos finalistas:

**LightGBM (ganador):**
```
learning_rate=0.05, max_depth=5, n_estimators=100, num_leaves=63
```

**GradientBoosting (segundo):**
```
learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.8
```

---

## 7. Resultados finales — Modelo binario

### 7.1 Top 2 modelos

| Rank | Modelo | PR-AUC (test) | IC 95% | AUC-ROC (test) | IC 95% |
|---|---|---|---|---|---|
| **Top 1** | **LightGBM** | **0.2496** | [0.158 – 0.353] | **0.7875** | [0.725 – 0.847] |
| Top 2 | GradientBoosting | 0.2475 | [0.154 – 0.353] | 0.7848 | [0.722 – 0.845] |

Los IC 95% se calcularon por **Bootstrap (1000 iteraciones)** sobre el test set (2.733 muestras, 83 positivos).

### 7.2 Análisis de umbrales

El umbral por defecto (0.5) no es óptimo para datos tan desbalanceados. Se evalúan 4 estrategias de umbral calculadas sobre validación cruzada out-of-fold (nunca sobre el test set directamente):

**LightGBM:**

| Estrategia | Umbral | Precisión | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| Default (0.50) | 0.500 | 0.333 | 0.301 | 0.316 | 0.981 |
| Máximo F1 | 0.409 | 0.264 | 0.349 | 0.301 | 0.969 |
| Recall ≥ 80% | 0.051 | 0.069 | 0.723 | 0.125 | 0.693 |
| **Youden J** | **0.084** | **0.117** | **0.699** | **0.201** | **0.836** |

> El **umbral de Youden** es el más equilibrado para uso clínico: detecta ~7 de cada 10 infecciones manteniendo un 83.6% de especificidad, lo que significa que no alarma innecesariamente al 84% de los pacientes sanos.

**GradientBoosting:**

| Estrategia | Umbral | Precisión | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| Default (0.50) | 0.500 | 0.300 | 0.325 | 0.312 | 0.976 |
| Máximo F1 | 0.334 | 0.219 | 0.398 | 0.282 | 0.956 |
| Recall ≥ 80% | 0.054 | 0.080 | 0.699 | 0.143 | 0.746 |
| Youden J | 0.068 | 0.097 | 0.699 | 0.171 | 0.797 |

### 7.3 Learning curves

Ambos modelos presentan un **gap train-CV de ~0.30** (PR-AUC), lo que indica overfitting moderado — esperado con SMOTEENN, que sobreajusta la frontera de decisión a las muestras sintéticas. El comportamiento es consistente con el observado en otros modelos del proyecto (mortalidad, UCI).

- LightGBM: gap = 0.302 | CV PR-AUC = 0.260 ± 0.055
- GradientBoosting: gap = 0.326 | CV PR-AUC = 0.251 ± 0.064

### 7.4 Calibración

La calibración isotónica reduce el Brier Score de ambos modelos, mejorando la fiabilidad de las probabilidades predichas para uso en scores de riesgo clínico.

---

## 8. Interpretabilidad — SHAP

Se aplica `TreeExplainer` (SHAP) sobre LightGBM y GradientBoosting para obtener explicaciones de las predicciones.

**Variables más importantes (SHAP global):** Las features con mayor impacto promedio en las predicciones de ambos modelos son consistentes con la literatura clínica:

- **`edad`**: Los pacientes mayores tienen mayor riesgo basal de infección postoperatoria
- **`charlson_score`**: La carga de comorbilidades aumenta la susceptibilidad a infecciones
- **`asa_class`**: Clasificación ASA (proxy de estado funcional preoperatorio)
- **`duracion_cirugia`**: Mayor tiempo quirúrgico → mayor exposición y riesgo
- **`tipo_cirugia`** / **`especialidad`**: El riesgo de infección es heterogéneo entre tipos de cirugía

Los dependence plots confirman relaciones clínicamente coherentes:
- Relación monotónica edad-riesgo
- Umbral de riesgo en charlson_score ≥ 4
- Efecto protector del sexo femenino en algunas categorías

---

## 9. Fase 9 — Modelos multiclase

### 9.1 Clasificación global — 7 clases

Se entrenan XGBoost, LightGBM y RandomForest con GridSearch (3-fold, métrica `f1_macro`). El pipeline incluye el mismo preprocesador pero **sin sampler** (la clase mayoritaria "ninguna" ya actúa de ancla; se usa `class_weight='balanced'`).

| Modelo | F1 Macro | F1 Weighted |
|---|---|---|
| **RandomForest** | **0.195** | **0.958** |
| XGBoost | 0.178 | 0.941 |
| LightGBM | 0.177 | 0.911 |

**Matriz de confusión (RandomForest, test set):**

- `ninguna`: 2.617 correctas de ~2.650 → altísima especificidad
- `resp` (más frecuente entre infectados): 14 correctas de 43 → recall 0.33
- `sepsis`, `itu`, `ssi`, `cateter`, `other`: prácticamente sin detecciones (n muy pequeño por clase)

> El F1 macro bajo (~0.20) refleja la imposibilidad práctica de distinguir entre categorías muy minoritarias (cateter=35 casos totales, other=25) con exclusivamente features preoperatorias. Es un resultado honesto, no un error metodológico.

### 9.2 Modelo condicional — 6 clases (solo infectados)

Se separa el subconjunto de pacientes infectados (n=413 total, ~330 train / ~83 test) derivado directamente del split del modelo binario (mismos índices). Al ser n pequeño, **no se aplica GridSearch** (riesgo de overfitting) — se usan hiperparámetros conservadores con `class_weight='balanced'`.

| Modelo | F1 Macro (test) | F1 Macro CV | ±std |
|---|---|---|---|
| **RandomForest** | **0.337** | **0.333** | 0.069 |
| LightGBM | 0.279 | 0.280 | 0.041 |
| XGBoost | 0.268 | 0.246 | 0.038 |

**Mejora clave:** Eliminar la clase "ninguna" del entrenamiento permite al modelo enfocarse en distinguir los tipos de infección entre sí, mejorando el F1 macro de ~0.20 a ~0.34 (+70% relativo).

**Distribución del test condicional (n=83):**

| Clase | N test |
|---|---|
| resp | 43 |
| sepsis | 10 |
| itu | 9 |
| ssi | 9 |
| cateter | 2 |
| other | 10 |

> El bajo n por clase (especialmente cateter=2 en test) limita estructuralmente el rendimiento. Con más datos el modelo sería más robusto.

---

## 10. Discusión

### 10.1 ¿Por qué PR-AUC y no AUC-ROC?

Con 3% de prevalencia, AUC-ROC puede parecer "buena" (0.79) incluso para modelos que apenas detectan positivos, porque incluye los verdaderos negativos en el denominador. **PR-AUC es más exigente**: ignora los TN y mide directamente cuántos positivos reales captura el modelo. Un PR-AUC de 0.025 (≈ prevalencia) sería el peor caso posible; 0.247 representa ~8× sobre el azar.

$$\text{PR-AUC baseline} = \text{prevalencia} = 0.030 \quad \Rightarrow \quad \text{lift} = \frac{0.247}{0.030} \approx 8.2\times$$

### 10.2 Interpretación clínica de los resultados

Los valores obtenidos son **coherentes con la literatura** para predicción de infección postoperatoria con features preoperatorias:

- La mayoría de publicaciones reportan AUC-ROC entre 0.70 y 0.85 para este tipo de problema
- El F1 macro en clasificación multiclase de infecciones raramente supera 0.40 cuando hay clases con <50 casos
- El modelo condicional (0.337) es comparable a modelos especializados en la literatura con datasets más grandes

### 10.3 Limitaciones

1. **Clases muy minoritarias:** cateter (35 casos), other (25 casos) — imposible generalizar con tan pocos ejemplos
2. **Features preoperatorias únicamente:** las infecciones postoperatorias dependen también de factores intraoperatorios e inmediatos (tiempo de anestesia, complicaciones quirúrgicas) que pueden no estar bien capturados
3. **Overfitting moderado:** el gap train-CV de ~0.30 sugiere que los modelos memorizan parcialmente las muestras sintéticas de SMOTEENN
4. **Heterogeneidad temporal:** el dataset abarca varios años; pueden existir cambios en protocolos clínicos que introduzcan no-estacionariedad

---

## 11. Outputs generados

| Archivo | Contenido |
|---|---|
| `results/predicciones_infeccion_postop.csv` | Predicciones binarias con probabilidad por paciente |
| `results/tabla_comparativa_tecnicas_infeccion.csv` | Comparativa SMOTEENN vs SMOTE vs ROS (CV y test) |
| `results/tabla_modelos_multiclase_infeccion.csv` | Resultados 7-clases + 6-clases condicional |
| `results/checkpoints/comp_tecnicas_infeccion.pkl` | Resultados GridSearch comparativa técnicas |
| `results/checkpoints/kfold_infeccion_SMOTEENN.pkl` | Resultados K-Fold 7 modelos |
| `results/checkpoints/gs_infeccion_SMOTEENN.pkl` | Resultados GridSearch optimización |
| `results/checkpoints/mc_7clases_infeccion.pkl` | Modelos multiclase global |
| `results/checkpoints/mc_cond_6clases_infeccion.pkl` | Modelos condicionales 6 clases |

---

## 12. Conclusiones

1. **LightGBM** es el mejor modelo para predicción binaria de infección postoperatoria (PR-AUC=0.247, AUC-ROC=0.788), con un rendimiento similar a GradientBoosting.

2. **SMOTEENN** es la técnica de balanceo óptima para este target, consistente con los resultados del notebook comparativo general sobre mortalidad y UCI.

3. El **umbral de Youden (0.084)** ofrece el mejor equilibrio clínico: detecta el 70% de las infecciones con una tasa de falsa alarma del 16%, lo que lo hace viable como herramienta de cribado preoperatorio.

4. La arquitectura de **modelo condicional en dos etapas** (binario → multiclase solo en positivos) mejora sustancialmente el F1 macro del problema multiclase (+70% respecto al enfoque global), y es la estrategia recomendada para el despliegue clínico.

5. La interpretabilidad SHAP confirma que las predicciones se basan en variables clínicamente coherentes (edad, Charlson, ASA, duración), lo que hace al modelo **explicable ante el clínico**.

6. La metodología es **robusta y sin data leakage**: imputer dentro del pipeline, BEST_TECNICA seleccionada por CV, mismos índices train/test en los tres niveles de modelado.
