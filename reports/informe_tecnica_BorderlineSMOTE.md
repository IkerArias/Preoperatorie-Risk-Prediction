# Informe Técnico — BorderlineSMOTE  
## Predicción de Mortalidad y Requerimiento de Cuidados Críticos Postquirúrgicos  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Técnica de balanceo:** BorderlineSMOTE (*Borderline Synthetic Minority Over-sampling Technique*)  
**Tipo:** Oversampling sintético restringido a la frontera de decisión  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Se aplica la técnica **BorderlineSMOTE** para abordar el severo desequilibrio de clases presente en el dataset clínico anestesiológico (mortalidad postquirúrgica: ~1.6%; requerimiento de cuidados críticos `outcome_intensivo`: ~2.04%). Se evalúan 7 algoritmos de clasificación a través de tres fases de validación (Baseline, K-Fold estratificado y GridSearch), usando PR-AUC como métrica de selección principal.

El mejor modelo para **mortalidad** es **GaussianNB** con PR-AUC = 0.1895, AUC-ROC = 0.8433 y Recall = 0.6486. Para **`outcome_intensivo`** (endpoint compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario), también **GaussianNB** con PR-AUC = 0.1792, AUC-ROC = 0.8322 y Recall = 0.4797. BorderlineSMOTE produce el **mayor AUC-ROC en mortalidad** (0.8433) entre las técnicas de oversampling puro y destaca por la limpieza de sus fronteras de decisión al excluir muestras ruidosas.

> **Nota metodológica:** El segundo target fue redefinido de `uci` (~0.8% prevalencia, 318 eventos) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (~2.04% prevalencia, 742 eventos) para corregir un sesgo de riesgo competitivo. Ver Sección 3.1.

---

## 1. Introducción

### 1.1 Motivación clínica

En la predicción postquirúrgica de mortalidad y necesidad de UCI, los modelos de machine learning entrenados sin balanceo de clases producen clasificadores degenerados: predicen siempre la clase negativa con ~98.4% de exactitud global, pero con recall = 0. Esta situación es clínicamente inaceptable, ya que deja sin detectar precisamente los casos que requieren intervención.

### 1.2 Hipótesis de BorderlineSMOTE

SMOTE estándar genera muestras sintéticas uniformemente distribuidas por todo el espacio de la clase positiva. Sin embargo, la mayoría de los errores de clasificación se concentran cerca de la **frontera de decisión**, no en el núcleo bien separado de la distribución positiva. BorderlineSMOTE parte de la hipótesis de que reforzar exclusivamente la zona fronteriza produce modelos con mejor discriminación en la región más crítica.

---

## 2. Marco Teórico — BorderlineSMOTE

### 2.1 Algoritmo

BorderlineSMOTE (Han et al., 2005) clasifica cada muestra positiva $x_i$ en tres categorías según su vecindad:

| Categoría | Condición | Tratamiento |
|---|---|---|
| **NOISE** | $n_{neg} = m$ (todos los vecinos son negativos) | Se descarta, no genera sintéticos |
| **DANGER (Borderline)** | $m/2 \leq n_{neg} < m$ (mayoría negativa) | **Genera sintéticos** |
| **SAFE** | $n_{neg} < m/2$ (mayoría positiva) | Se omite, no necesita refuerzo |

Solo las muestras clasificadas como **DANGER** participan en la generación de sintéticos. Los sintéticos se crean interpolando entre la muestra DANGER y sus vecinos positivos más cercanos.

**Pasos del algoritmo:**

1. Para cada $x_i \in C_{min}$, encontrar sus $m$ vecinos más cercanos (de ambas clases).
2. Calcular $n_{neg}^{(i)}$ = número de vecinos de clase mayoritaria.
3. Clasificar según la tabla anterior.
4. Para cada muestra DANGER $x_i$: encontrar sus $k$ vecinos más cercanos **dentro de la clase positiva** $\hat{x}_j^{(i)}$ y generar $x_{nuevo} = x_i + \lambda \cdot (\hat{x}_j^{(i)} - x_i)$ con $\lambda \sim U(0, 1)$.

### 2.2 Formulación matemática

$$x_i \text{ es DANGER} \iff \left\lfloor \frac{m}{2} \right\rfloor \leq n_{neg}^{(i)} < m$$

$$x_{nuevo} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim \mathcal{U}(0, 1), \quad x_{nn} \in \text{k-vecinos positivos}$$

Donde $m = 10$ (vecinos para clasificación) y $k = 5$ (vecinos para interpolación) son los parámetros por defecto en imbalanced-learn.

### 2.3 Parámetro `sampling_strategy = 0.3`

Con `sampling_strategy = 0.3` se genera sobremuestra hasta alcanzar $n_{pos}/n_{neg} = 0.3$. Solo las muestras DANGER contribuyen a los sintéticos, lo que en la práctica significa que el oversampling es más conservador que en SMOTE estándar: no todas las positivas participan.

### 2.4 BorderlineSMOTE vs SMOTE vs ADASYN

| Característica | SMOTE | ADASYN | BorderlineSMOTE |
|---|---|---|---|
| ¿Qué muestras generan sintéticos? | Todas las positivas | Todas, ponderado por $\hat{d}_i$ | Solo las DANGER (borderline) |
| Región de generación | Todo el espacio positivo | Concentrada en frontera | **Estrictamente en la frontera** |
| Tratamiento de NOISE | No existe | No existe | **Excluye NOISE** |
| Garantía de coherencia | No exige positivos vecinos | No exige positivos vecinos | Interpola con vecinos positivos |
| Conservadurismo | Bajo | Medio | **Alto** |

### 2.5 Dos variantes: BorderlineSMOTE-1 y BorderlineSMOTE-2

- **BorderlineSMOTE-1**: interpola entre la muestra DANGER y sus vecinos positivos.
- **BorderlineSMOTE-2**: interpola con cualquier vecino (positivo o negativo), pero ponderando con $\lambda \sim \mathcal{U}(0, 0.5)$ si el vecino es negativo (sintético "más cercano" a la frontera).

La implementación por defecto de imbalanced-learn es la variante 1.

### 2.6 Ventajas y limitaciones

**Ventajas:**
- Focaliza la capacidad de oversampling en la frontera de decisión, donde los errores son más costosos
- Excluye el NOISE (outliers extremos), reduciendo el riesgo de contaminación por ruido
- Produce fronteras de decisión más nítidas que SMOTE estándar al no reforzar el kernel denso

**Limitaciones:**
- Si pocas muestras positivas son clasificadas como DANGER, se generan pocos sintéticos y el beneficio es limitado
- Depende de la calidad del vecindario $k$-NN: si el espacio de features no es euclidean-coherente, la clasificación SAFE/DANGER/NOISE puede ser errónea
- Puede ignorar regiones positivas importantes no representadas en la frontera

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

El target original `uci` sufría un **problema de riesgo competitivo** (Fine & Gray, 1999): un paciente que fallece antes de ser trasladado a UCI queda etiquetado como negativo, produciendo una subestimación sistemática de eventos graves. El endpoint compuesto `outcome_intensivo = uci OR mortalidad` corrige este sesgo: (1) **clínicamente**, la acción ante UCI o muerte inminente es idéntica; (2) **estadísticamente**, 742 eventos producen estimaciones más estables; (3) las guías CONSORT recomiendan endpoints compuestos cuando los eventos comparten el mismo mecanismo fisiopatológico. La variable `mortalidad` se **elimina de las features** para evitar data leakage.

### 3.2 Integridad del test set

Cuatro barreras anti-leakage garantizan la validez del experimento: (1) split previo a cualquier procesado, (2) GroupShuffleSplit por paciente, (3) BorderlineSMOTE aplicado solo dentro del pipeline (no ve X_test), (4) StratifiedGroupKFold mantiene la prevalencia real en cada fold.

### 3.3 Pipeline

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE

pipeline = ImbPipeline([
    ('preprocessor',  preprocessor),       # StandardScaler en 'edad'
    ('sampler',       BorderlineSMOTE(sampling_strategy=0.3,
                                      random_state=42,
                                      kind='borderline-1')),
    ('model',         classifier)
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
| **Fase 2 — K-Fold** | StratifiedGroupKFold, 5 folds | Robustez y generalización |
| **Fase 3 — GridSearch** | GridSearchCV(scoring='average_precision') | Optimización de hiperparámetros |

---

## 4. Resultados

### 4.1 Fase 2 — K-Fold (robustez)

| Target | Modelo | KF PR-AUC (media) | KF PR-AUC (std) | CV | Interpretación |
|---|---|---|---|---|---|
| MORTALIDAD | GaussianNB | **0.1814** | ±0.0119 | 6.56% | Estabilidad aceptable |
| outcome_intensivo | GaussianNB | **0.2147** | ±0.0150 | 6.99% | Buena estabilidad |

La variabilidad de `outcome_intensivo` (±0.0150) es significativamente menor que la del antiguo target `uci` (±0.0331), confirmando que el mayor número de eventos (742 vs 318) del endpoint compuesto produce estimaciones por fold más estables.

### 4.2 Fase 3 — GridSearch, Mortalidad

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision | Specificity |
|---|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1895** | 0.8433 | 0.6486 | 0.1173 | 0.0645 | 0.8456 |
| 2 | XGBoost | 0.1375 | 0.8916 | 0.4805 | 0.1955 | 0.1227 | 0.9437 |
| 3 | LightGBM | 0.1314 | 0.8912 | 0.4925 | 0.2001 | 0.1256 | 0.9438 |
| 4 | GradientBoosting | 0.1309 | 0.8913 | 0.4805 | 0.1970 | 0.1239 | 0.9443 |
| 5 | RandomForest | 0.1303 | 0.8932 | 0.3063 | 0.2010 | 0.1496 | 0.9714 |
| 6 | ExtraTrees | 0.1257 | 0.8807 | 0.2222 | 0.1805 | 0.1520 | 0.9797 |
| 7 | LogisticRegression | 0.1161 | 0.8796 | 0.5075 | 0.1782 | 0.1081 | 0.9313 |

**Hiperparámetros óptimos (GaussianNB — Mortalidad):** `var_smoothing = 1e-6`

### 4.3 Fase 3 — GridSearch, `outcome_intensivo`

Target compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario. Prevalencia en test: 2.04% (148/7.267 episodios).

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1792** | 0.8322 | 0.4797 | 0.1336 | 0.0776 |
| 2 | XGBoost | 0.1357 | 0.8738 | 0.3919 | 0.2086 | 0.1526 |
| 3 | LightGBM | 0.1345 | 0.8761 | 0.3919 | 0.2094 | 0.1535 |
| 4 | GradientBoosting | 0.1310 | 0.8721 | 0.3851 | 0.2096 | 0.1545 |
| 5 | RandomForest | 0.1237 | 0.8896 | 0.2432 | 0.1885 | 0.1491 |
| 6 | LogisticRegression | 0.1216 | 0.8708 | 0.4257 | 0.1886 | 0.1234 |
| 7 | ExtraTrees | 0.1197 | 0.8830 | 0.3108 | 0.2155 | 0.1689 |

**Hiperparámetros óptimos (GaussianNB — `outcome_intensivo`):** `var_smoothing = 1e-5`

> **Interpretación:** Con el nuevo target `outcome_intensivo` (2.04% prevalencia), el ranking de técnicas cambia respecto al antiguo `uci`: los números son más estables y las diferencias entre modelos más fiables. GaussianNB lidera en PR-AUC aunque con precisión baja (7.8%); XGBoost y LightGBM ofrecen mejor equilibrio (F1~0.21) siendo preferibles en despliegue clínico.

---

## 5. Análisis del Mejor Modelo — GaussianNB + BorderlineSMOTE

### 5.1 Métricas clínicas interpretadas

| Métrica | Mortalidad | UCI | Interpretación clínica |
|---|---|---|---|
| PR-AUC | 0.1895 | 0.1792 | 11.8× y 8.8× sobre el azar (prevalencias ~1.6% y ~2.04%) |
| AUC-ROC | 0.8433 | 0.8322 | El mayor AUC-ROC en mortalidad de todas las técnicas puras (excl. SMOTEENN) |
| Recall | 0.6486 | 0.4797 | 65% de muertes detectadas; 48% de eventos críticos detectados |
| Precision | 0.0645 | 0.0776 | 1 de cada 15.5 alarmas de muerte; 1 de cada 13 de outcome_intensivo |
| F1 | 0.1173 | 0.1336 | Bajo por prevalencia; no es la métrica adecuada aquí |

### 5.2 Interpretación del alto AUC-ROC en mortalidad

BorderlineSMOTE produce el mayor AUC-ROC en mortalidad (0.8433) entre todas las técnicas de oversampling puro. Esto puede explicarse por el mecanismo de exclusión de NOISE: al no generar sintéticos en zonas extremas, los modelos entrenados tienen fronteras de decisión más "limpias" que se benefician de los verdaderos negativos (TN), componente central del AUC-ROC.

### 5.3 PR-AUC inferior a SMOTE y ADASYN en mortalidad

BorderlineSMOTE obtiene PR-AUC=0.1895 vs SMOTE=0.2012 (−0.0117) en mortalidad, y PR-AUC=0.1792 vs SMOTE=0.1790 (−0.0002) en `outcome_intensivo`. Una explicación es que, en este dataset, el número de muestras DANGER puede ser limitado (si muchos positivos están en zonas bien separadas), reduciendo la diversidad de sintéticos. Para `outcome_intensivo`, sin embargo, ambas técnicas son prácticamente equivalentes, indicando que la ponderación fronteriza no aporta ventaja adicional con 742 eventos.

---

## 6. Análisis del Umbral de Decisión

| Umbral | Descripción | Recall | Precision | VPP |
|---|---|---|---|---|
| 0.5 (defecto) | Corte estándar | ~0.00 | N/A | N/A |
| F1-max | Balance precision/recall | ~0.65 | ~0.06 | 6% |
| Recall ≥ 90% | Screening clínico | ≥0.90 | ~0.03 | 3% |

Para UCI, el umbral F1-max produce recall ~0.83 con precision ~0.05, ofreciendo un equilibrio aceptable para sistemas de alerta temprana.

---

## 7. Calibración Probabilística

Con BorderlineSMOTE, las probabilidades de salida están infladas (el modelo "cree" que la prevalencia es ~23%). La calibración con **Isotonic Regression** (o Platt scaling) ajusta estas probabilidades sin alterar el ranking. Esto es especialmente importante para sistemas clínicos donde la probabilidad absoluta informa decisiones de triaje.

---

## 8. Conclusiones

1. **BorderlineSMOTE + GaussianNB** ofrece PR-AUC = 0.1895 (mortalidad) y 0.1792 (`outcome_intensivo`), con el mayor AUC-ROC para mortalidad (0.8433) entre las técnicas de oversampling puro.

2. **El endpoint compuesto `outcome_intensivo` resuelve el sesgo de riesgo competitivo** del antiguo target `uci`: los 742 eventos producen estimaciones K-Fold más estables (±0.015 vs ±0.033 anteriores) y resultados más fiables.

3. La **exclusión de muestras NOISE** es la característica diferenciadora de BorderlineSMOTE: produce fronteras de decisión más limpias (mayor AUC-ROC) a costa de menor cobertura sintética.

4. La **robustez K-Fold** (±0.0119 mortalidad, ±0.0150 `outcome_intensivo`) es comparable a SMOTE, indicando que la restricción a muestras borderline no introduce inestabilidad en la validación cruzada.

5. **Recomendación de uso**: BorderlineSMOTE es preferible cuando el objetivo clínico es maximizar la discriminación global (AUC-ROC) y se acepta un ligero sacrificio en PR-AUC. Para maximizar PR-AUC, SMOTEENN es superior.

---

## Referencias

- Han, H., Wang, W. Y., & Mao, B. H. (2005). Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning. *ICIC 2005*, LNCS 3644, 878-887.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling TEchnique. *JAIR*, 16, 321-357.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10(3), e0118432.
- Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn. *JMLR*, 18(17), 1-5.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
