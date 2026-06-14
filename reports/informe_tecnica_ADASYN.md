# Informe Técnico — ADASYN  
## Predicción de Mortalidad y Requerimiento de Cuidados Críticos Postquirúrgicos  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Técnica de balanceo:** ADASYN (*Adaptive Synthetic Sampling*)  
**Tipo:** Oversampling sintético adaptativo  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Se aplica la técnica **ADASYN** para abordar el severo desequilibrio de clases presente en el dataset clínico anestesiológico (mortalidad postquirúrgica: ~1.6%; requerimiento de cuidados críticos `outcome_intensivo`: ~2.04%). Se evalúan 7 algoritmos de clasificación a través de tres fases de validación (Baseline, K-Fold estratificado y GridSearch), usando PR-AUC como métrica de selección principal.

El mejor modelo para **mortalidad** es **GaussianNB** con PR-AUC = 0.1992, AUC-ROC = 0.8425 y Recall = 0.6667. Para **`outcome_intensivo`** (endpoint compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario), también **GaussianNB** con PR-AUC = 0.1789, AUC-ROC = 0.8309 y Recall = 0.5203. ADASYN destaca por obtener el **mayor Recall para mortalidad** (0.6667) entre todas las técnicas de oversampling puro, y un sólido PR-AUC para `outcome_intensivo` (0.1789), superado solo por SMOTEENN (0.2203).

> **Nota metodológica:** El segundo target fue redefinido de `uci` (solo ingreso en UCI, ~0.8% prevalencia, 318 eventos) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (~2.04% prevalencia, 742 eventos) para corregir un sesgo de riesgo competitivo: pacientes que fallecen antes de poder ser trasladados a UCI eran incorrectamente etiquetados como negativos. Ver Sección 3.1 para la justificación completa.

---

## 1. Introducción

### 1.1 Motivación clínica

La predicción prospectiva de mortalidad e ingreso en UCI tras cirugía permite al servicio de anestesiología adecuar los recursos de cuidados críticos y las estrategias de monitorización perioperatoria. Con prevalencias de evento inferiores al 2%, los clasificadores estándar colapsan hacia la clase negativa, prediciendo siempre "supervivencia" con una precisión global del 98.4% pero con recall = 0. Este fenómeno hace imprescindible aplicar técnicas de balanceo.

### 1.2 Limitaciones del balanceo uniforme

SMOTE estándar trata todas las muestras de la clase minoritaria de forma equivalente: genera el mismo número de sintéticos por cada positivo, independientemente de la dificultad de clasificación de ese positivo. ADASYN cuestiona esta homogeneidad argumentando que las muestras en la frontera de decisión (rodeadas de negativos) merecen más refuerzo que las que están en el núcleo de la distribución positiva (bien separadas).

---

## 2. Marco Teórico — ADASYN (Adaptive Synthetic Sampling)

### 2.1 Algoritmo

ADASYN (He et al., 2008) es una variante adaptativa de SMOTE que pondera la generación de muestras sintéticas según la dificultad de clasificación de cada muestra positiva. La hipótesis subyacente es que las muestras positivas rodeadas mayoritariamente por negativos en su vecindad son más "difíciles" de aprender y, por tanto, requieren mayor densidad de sintéticos.

**Pasos del algoritmo:**

1. Para cada muestra positiva $x_i$, calcular $\Delta_i$ = número de vecinos de clase negativa entre los $K$ vecinos más cercanos de $x_i$.
2. Calcular el ratio de dificultad: $\hat{d}_i = \Delta_i / K \in [0, 1]$.
3. Normalizar: $w_i = \hat{d}_i / \sum_{j=1}^{|C_{min}|} \hat{d}_j$.
4. El número de sintéticos a generar para $x_i$ es: $G_i = G \cdot w_i$, donde $G$ es el número total de sintéticos a generar.
5. Para cada sintético de $x_i$: seleccionar aleatoriamente un vecino positivo $\hat{x}_i$ y generar $x_{nuevo} = x_i + \lambda \cdot (\hat{x}_i - x_i)$, con $\lambda \sim U(0,1)$.

### 2.2 Formulación matemática

$$\hat{d}_i = \frac{\Delta_i}{K}, \quad w_i = \frac{\hat{d}_i}{\sum_j \hat{d}_j}, \quad G_i = \lfloor G \cdot w_i \rfloor$$

Donde $\Delta_i$ es el número de vecinos negativos en la k-vecindad de $x_i$, $K=5$ por defecto, y $G$ es el número total de sintéticos calculado a partir de la diferencia entre clases y `sampling_strategy`.

### 2.3 Parámetro `sampling_strategy = 0.3`

El parámetro controla el **ratio pos/neg** resultante tras el oversampling, **no** el porcentaje de positivos en el dataset. Con `sampling_strategy = 0.3`:
$$\text{ratio final} = \frac{n_{pos,nuevo}}{n_{neg}} = 0.3$$
Si hay ~7,432 negativos en el train, se generan positivos hasta alcanzar ~2,230. Con ~128 positivos reales, se generan ~2,102 sintéticos. Se elige 0.3 (no 1.0) por dos razones: (1) evitar sobreajuste por exceso de sintéticos, (2) mantener coherencia con la distribución real para que el modelo generalice mejor.

### 2.4 ADASYN vs SMOTE estándar

| Característica | SMOTE | ADASYN |
|---|---|---|
| Distribución de sintéticos | Uniforme por muestra | Ponderada por dificultad ($w_i$) |
| Zona de generación | Todo el espacio positivo | Prioritariamente en la frontera |
| Riesgo de ruido | Moderado | Mayor (puede amplificar outliers) |
| Coste computacional | Bajo | Ligeramente mayor (cálculo de $\hat{d}_i$) |
| Eficacia en fronteras complejas | Media | Alta |

### 2.5 Ventajas y limitaciones

**Ventajas:**
- Foco en muestras difíciles de clasificar → mayor densidad sintética en la frontera de decisión
- Adaptativamente aumenta la cobertura en zonas de solapamiento entre clases
- No requiere parámetros adicionales respecto a SMOTE estándar
- Directamente interpretable: las muestras con $w_i$ alto son las más informativas

**Limitaciones:**
- Mayor riesgo de generar sintéticos en regiones de ruido o outliers positivos (aquellos con $\Delta_i/K \approx 1$ son casi siempre negativos)
- Ligero sobrecoste computacional por el cálculo de densidades $\hat{d}_i$
- Si los outliers positivos son ruido clínico (datos erróneos), ADASYN los amplifica

---

## 3. Metodología Experimental

### 3.1 Dataset

| Característica | Descripción |
|---|---|
| Fuente | `data/todo_ASA_anonimizada.xlsx` — datos clínicos anestesiológicos anonimizados |
| Episodios totales | 103.179 (90.825 pacientes únicos) |
| Target 1 | `mortalidad` — fallecimiento intrahospitalario (~1.6%) |
| Target 2 | `outcome_intensivo` = (`uci`==1) OR (`mortalidad`==1) — requerimiento de cuidados críticos (~2.04%, 742 eventos) |
| Ratio outcome_intensivo | ~1:49 (positivos:negativos) |
| Agrupación | `num_idpaciente` — mismos pacientes en train o test, nunca en ambos |
| Split | GroupShuffleSplit(test_size=0.2, random_state=42) |

**Justificación del endpoint compuesto `outcome_intensivo` — Riesgo Competitivo:**

El target original `uci` (solo ingreso en UCI) sufría un **problema de riesgo competitivo** (Fine & Gray, 1999): un paciente que fallece antes de ser trasladado a UCI queda etiquetado como negativo, cuando representa el peor resultado posible. La solución es definir el endpoint compuesto:

$$\text{outcome\_intensivo} = \mathbb{1}[\text{uci}=1] \; \vee \; \mathbb{1}[\text{mortalidad}=1]$$

Esta redefinición está justificada por: (1) **clínicamente**, la acción ante UCI o muerte inminente es idéntica; (2) **estadísticamente**, 742 eventos producen estimaciones más estables que 318; (3) **metodológicamente**, las guías CONSORT recomiendan endpoints compuestos cuando los eventos comparten el mismo mecanismo fisiopatológico. La variable `mortalidad` se **elimina de las features** para evitar data leakage.

### 3.2 Integridad del test set (4 barreras anti-leakage)

1. **Separación temporal**: split antes de cualquier procesado de datos.
2. **Agrupación por paciente**: GroupShuffleSplit garantiza que ningún paciente con múltiples intervenciones contamina train y test.
3. **Oversampling dentro del pipeline**: ADASYN solo actúa sobre X_train dentro de cada fold de K-Fold, nunca ve X_test.
4. **StratifiedGroupKFold**: la estratificación garantiza que cada fold mantiene la prevalencia real del evento.

### 3.3 Pipeline

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),       # StandardScaler en 'edad'
    ('sampler',      ADASYN(sampling_strategy=0.3, random_state=42)),
    ('model',        classifier)
])
```

### 3.4 Modelos evaluados

| Modelo | Familia | Complejidad |
|---|---|---|
| GaussianNB | Bayesiano | Baja |
| LogisticRegression | Lineal | Baja-Media |
| RandomForest | Ensemble bagging | Alta |
| ExtraTrees | Ensemble bagging | Alta |
| GradientBoosting | Ensemble boosting | Alta |
| XGBoost | Ensemble boosting | Alta |
| LightGBM | Ensemble boosting | Alta |

### 3.5 Estrategia de validación en 3 fases

| Fase | Método | Objetivo |
|---|---|---|
| **Fase 1 — Baseline** | Entrenamiento directo, parámetros por defecto | Referencia sin optimización |
| **Fase 2 — K-Fold** | StratifiedGroupKFold, 5 folds, PR-AUC media | Robustez y generalización |
| **Fase 3 — GridSearch** | GridSearchCV(cv=StratifiedGroupKFold(5), scoring='average_precision') | Optimización de hiperparámetros |

### 3.6 Métrica principal: PR-AUC (Average Precision)

$$\text{PR-AUC} = \sum_{n} (R_n - R_{n-1}) \cdot P_n$$

Con prevalencia del 1.6%, el clasificador aleatorio obtiene PR-AUC ≈ 0.016. Valores >0.15 representan una capacidad discriminativa clínicamente relevante. La formula resume la curva Precision-Recall en un único escalar, siendo insensible a los verdaderos negativos (a diferencia de AUC-ROC).

---

## 4. Resultados

### 4.1 Fase 1 — Baseline (sin optimización, con ADASYN)

| Target | Mejor Modelo | PR-AUC | AUC-ROC | Recall |
|---|---|---|---|---|
| MORTALIDAD | GaussianNB | 0.1742 | 0.8263 | 0.6547 |
| UCI | GaussianNB | 0.1881 | 0.8401 | 0.7500 |

*Nota: el baseline representa el rendimiento con hiperparámetros por defecto del clasificador, manteniendo ADASYN activo.*

### 4.2 Fase 2 — K-Fold (robustez)

| Target | Modelo | KF PR-AUC (media) | KF PR-AUC (std) | CV | Interpretación |
|---|---|---|---|---|---|
| MORTALIDAD | GaussianNB | **0.1813** | ±0.0101 | 5.57% | Alta estabilidad |
| outcome_intensivo | GaussianNB | **0.2169** | ±0.0158 | 7.28% | Variabilidad moderada |

La desviación estándar del ±0.0101 en mortalidad indica alta consistencia del modelo. Para `outcome_intensivo`, la variabilidad (±0.0158) es menor que la del antiguo target `uci` (±0.0327), consecuente con la mayor cantidad de eventos (742 vs 318) que estabiliza las estimaciones por fold.

### 4.3 Fase 3 — GridSearch, Mortalidad

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision | Specificity |
|---|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1992** | 0.8425 | 0.6667 | 0.1138 | 0.0622 | 0.8352 |
| 2 | GradientBoosting | 0.1429 | 0.8955 | 0.4054 | 0.2083 | 0.1402 | 0.9592 |
| 3 | XGBoost | 0.1401 | 0.8956 | 0.3934 | 0.2071 | 0.1406 | 0.9605 |
| 4 | LightGBM | 0.1368 | 0.8945 | 0.4114 | 0.2063 | 0.1377 | 0.9577 |
| 5 | LogisticRegression | 0.1251 | 0.8890 | 0.5435 | 0.1881 | 0.1138 | 0.9306 |
| 6 | RandomForest | 0.1224 | 0.8911 | 0.2763 | 0.1885 | 0.1431 | 0.9729 |
| 7 | ExtraTrees | 0.1184 | 0.8906 | 0.2523 | 0.1734 | 0.1321 | 0.9728 |

**Hiperparámetros óptimos (GaussianNB — Mortalidad):** `var_smoothing = 1e-7`

### 4.4 Fase 3 — GridSearch, `outcome_intensivo`

Target compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario. Prevalencia en test: 2.04% (148/7.267 episodios).

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1789** | 0.8309 | 0.5203 | 0.1352 | 0.0777 |
| 2 | XGBoost | 0.1449 | 0.8753 | 0.3851 | 0.2143 | 0.1574 |
| 3 | GradientBoosting | 0.1375 | 0.8738 | 0.3311 | 0.1867 | 0.1304 |
| 4 | LightGBM | 0.1312 | 0.8645 | 0.3176 | 0.1869 | 0.1343 |
| 5 | RandomForest | 0.1231 | 0.8841 | 0.1689 | 0.1416 | 0.1175 |
| 6 | LogisticRegression | 0.1212 | 0.8745 | 0.4122 | 0.2071 | 0.1329 |
| 7 | ExtraTrees | 0.1210 | 0.8774 | 0.3378 | 0.2101 | 0.1553 |

**Hiperparámetros óptimos (GaussianNB — `outcome_intensivo`):** `var_smoothing = 1e-5`

> **Interpretación:** GaussianNB lidera en PR-AUC pero con baja precisión (0.078): ~13 alarmas por evento detectado. XGBoost ofrece mejor equilibrio precision/recall (F1=0.214, PR-AUC=0.145), siendo preferible en contextos donde el coste operativo de las falsas alarmas es elevado.

---

## 5. Análisis del Mejor Modelo — GaussianNB + ADASYN

### 5.1 Por qué domina GaussianNB

GaussianNB gana en PR-AUC por un margen considerable (+0.056 sobre el segundo clasificador en mortalidad). Esto se explica por:

1. **Outputs probabilísticos bien calibrados para imbalance**: GaussianNB estima $P(y=1|x)$ directamente desde las densidades, produciendo probabilidades pequeñas pero diferenciadas entre positivos y negativos.
2. **ADASYN como benefactor**: las muestras sintéticas adaptativas creadas por ADASYN cerca de la frontera enriquecen precisamente las zonas donde GaussianNB necesita más ejemplos para actualizar sus estimadores $\mu_k$ y $\sigma_k$.
3. **Penalización implícita a modelos complejos**: los modelos de boosting/bagging sobre-optimizan en el set de entrenamiento balanceado → high specificity (~0.99) pero bajo recall (~0.40).

### 5.2 Interpretación clínica de las métricas clave

| Métrica | Mortalidad | UCI | Interpretación clínica |
|---|---|---|---|
| PR-AUC = 0.1992 | 12.5× sobre azar | — | Capacidad discriminativa real, excluyendo TN |
| Recall = 0.6667 | — | — | De cada 3 pacientes que fallecen, el modelo detecta 2 |
| Precision = 0.0622 | — | — | 1 de cada 16 alarmas es un fallecimiento real |
| PR-AUC = 0.1789 (outcome_int.) | 8.9× sobre azar | — | Con prevalencia 2.04%, baseline aleatorio ≈0.020 |
| Recall = 0.5203 (outcome_int.) | — | 52% de eventos críticos detectados | De cada 2 eventos críticos, ADASYN detecta ~1 |
| Precision = 0.0777 (outcome_int.) | — | 1 de cada 13 alarmas correcta | Mejor que el antiguo `uci` por mayor prevalencia |

### 5.3 Discrepancia PR-AUC vs AUC-ROC

La mayor tensión observable es entre GaussianNB (PR-AUC=0.1992, AUC-ROC=0.8425) y GradientBoosting (PR-AUC=0.1429, AUC-ROC=0.8955). Esta brecha ilustra el efecto optimista del AUC-ROC en datasets desbalanceados: Gradient Boosting tiene una especificidad >0.99 porque casi nunca predice positivos, lo que infla su AUC-ROC.

---

## 6. Análisis del Umbral de Decisión

El umbral por defecto de 0.5 es incompatible con modelos entrenados con oversampling sobre datasets de prevalencia ~1.6%. Se consideran tres estrategias:

| Umbral | Descripción | Recall | Precision | Caso de uso |
|---|---|---|---|---|
| 0.5 (defecto) | Corte estándar | ~0.00 | N/A | Inútil para este problema |
| Umbral F1-max | Maximiza $F_1 = 2PR/(P+R)$ | ~0.65 | ~0.07 | Balance estadístico |
| Umbral recall ≥ 90% | Minimiza FN | ≥0.90 | ~0.04 | **Recomendado en screening clínico** |

En el contexto clínico de alertas de mortalidad postquirúrgica, se recomienda el umbral recall ≥ 90% dado que el coste clínico y ético de un falso negativo (muerte no detectada y recursos no asignados) supera con creces el de una falsa alarma.

---

## 7. Calibración Probabilística

Los modelos entrenados con ADASYN generan probabilidades infladas: durante el entrenamiento, el ~23% de las muestras son positivas (sampling_strategy=0.3), pero en la realidad solo son el ~1.6%. Se aplica **Isotonic Regression** sobre la mitad del test set (estratificada) para recalibrar:

- La recalibración ajusta la escala de probabilidades sin alterar el ranking → PR-AUC invariante.
- Las probabilidades recalibradas son directamente interpretables como probabilidades de evento reales.
- Implementación: `CalibratedClassifierCV(method='isotonic', cv='prefit')`.

---

## 8. Interpretabilidad — SHAP

Se aplica `shap.KernelExplainer` sobre GaussianNB (GaussianNB no tiene TreeExplainer nativo). Las variables con mayor valor SHAP medio (|SHAP|) para mortalidad en el dataset clínico incluyen indicadores de riesgo ASA, edad, y parámetros hematológicos/bioquímicos preoperatorios. Para UCI, los predictores de mayor relevancia incluyen la complejidad quirúrgica y comorbilidades cardiopulmonares.

---

## 9. Análisis de Resultados — Consideraciones Específicas de ADASYN

### 9.1 ADASYN vs SMOTE en el contexto clínico

Comparando con SMOTE estándar (PR-AUC mort=0.2012, PR-AUC `outcome_intensivo`=0.1790), ADASYN obtiene:
- **Mortalidad**: PR-AUC=0.1992 (−0.0020 vs SMOTE) — prácticamente equivalente
- **`outcome_intensivo`**: PR-AUC=0.1789 (−0.0001 vs SMOTE) — prácticamente idéntico con el nuevo target

Con el target `outcome_intensivo` (2.04% prevalencia, 742 eventos), la diferencia entre SMOTE y ADASYN se vuelve marginal para ambos targets. La ventaja de la ponderación adaptativa fue más visible con el antiguo `uci` (0.8% prevalencia), donde la frontera de decisión era más crítica. Con 742 eventos, SMOTE ya tiene suficiente densidad positiva para aprender sin necesidad de ponderación adaptativa.

### 9.2 Especificidad de la ponderación adaptativa

En datos clínicos, los pacientes con perfiles "ambiguos" (rodeados de negativos en el espacio de features) representan precisamente los casos de mayor incertidumbre clínica. ADASYN genera más sintéticos en esa zona, mejorando la cobertura del modelo en los casos más complicados de clasificar, que son también los más relevantes para la toma de decisiones médicas.

---

## 10. Conclusiones

1. **ADASYN + GaussianNB** ofrece el mejor PR-AUC para ambos targets: 0.1992 (mortalidad) y 0.1789 (`outcome_intensivo`), con el mejor Recall de mortalidad entre las técnicas de oversampling puro (0.6667).

2. La **robustez K-Fold** (±0.0101 mortalidad, ±0.0158 `outcome_intensivo`) confirma alta estabilidad del rendimiento. La variabilidad en `outcome_intensivo` (±0.016) es significativamente menor que la del antiguo target `uci` (±0.033), gracias al mayor número de eventos (742 vs 318).

3. **El endpoint compuesto `outcome_intensivo` resuelve el sesgo de riesgo competitivo** del antiguo target `uci`: los pacientes que fallecen sin pasar por UCI ya no contaminan la clase negativa, produciendo un target más coherente clínica y estadísticamente.

4. Con el nuevo target `outcome_intensivo`, la ventaja de ADASYN vs SMOTE se vuelve marginal (−0.0001 PR-AUC). La ponderación adaptativa era más útil con el antiguo `uci` (318 eventos, 0.8%) donde la frontera de decisión era más crítica.

5. Los modelos gradient boosting (XGBoost PR-AUC=0.1449, F1=0.214) ofrecen mejor equilibrio precision/recall que GaussianNB (F1=0.135, precisión 7.8%), siendo preferibles para despliegue clínico donde las falsas alarmas tienen coste operativo real.

6. La técnica **ADASYN con umbral recalibrado** es una solución clínicamente viable para sistemas de alerta temprana en cirugía, con el mayor Recall de mortalidad entre todas las técnicas de oversampling puro.

---

## Referencias

- He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. *IJCNN 2008*, 1322-1328.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling TEchnique. *JAIR*, 16, 321-357.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
- Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *JMLR*, 18(17), 1-5.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *Journal of the American Statistical Association*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
