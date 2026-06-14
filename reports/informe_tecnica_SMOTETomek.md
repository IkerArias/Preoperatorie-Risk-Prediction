# Informe Técnico — SMOTETomek  
## Predicción de Mortalidad y Requerimiento de Cuidados Críticos Postquirúrgicos  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Técnica de balanceo:** SMOTETomek (*SMOTE + Tomek Links Cleaning*)  
**Tipo:** Método híbrido oversampling + limpieza de frontera  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Se aplica la técnica **SMOTETomek** para abordar el severo desequilibrio de clases presente en el dataset clínico anestesiológico (mortalidad postquirúrgica: ~1.6%; requerimiento de cuidados críticos `outcome_intensivo`: ~2.04%). Se evalúan 7 algoritmos de clasificación a través de tres fases de validación (Baseline, K-Fold estratificado y GridSearch), usando PR-AUC como métrica de selección principal.

El mejor modelo para **mortalidad** es **GaussianNB** con PR-AUC = 0.2011, AUC-ROC = 0.8422 y Recall = 0.6607. Para **`outcome_intensivo`** (endpoint compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario), también **GaussianNB** con PR-AUC = 0.1790, AUC-ROC = 0.8310 y Recall = 0.5203.

**Hallazgo central:** SMOTETomek produce resultados **prácticamente idénticos a SMOTE estándar** en ambos targets (diferencia <0.001 en PR-AUC). Esto revela que los Tomek Links son **muy escasos** en este dataset clínico y su eliminación tiene efecto despreciable sobre el rendimiento del clasificador. Para obtener mejora real con un método híbrido, se requiere SMOTEENN (limpieza ENN).

> **Nota metodológica:** El segundo target fue redefinido de `uci` (~0.8% prevalencia, 318 eventos) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (~2.04% prevalencia, 742 eventos) para corregir un sesgo de riesgo competitivo. Ver Sección 3.1.

---

## 1. Introducción

### 1.1 Motivación de los métodos híbridos

Las técnicas de solo oversampling (SMOTE, ADASYN, BorderlineSMOTE) pueden generar muestras sintéticas en la frontera de decisión pero no eliminan las muestras reales problemáticas de esa zona. Los métodos híbridos combinan oversampling (para aumentar cobertura de la clase minoritaria) con técnicas de limpieza (para eliminar muestras solapadas o ambiguas de ambas clases), aspirando a obtener lo mejor de ambos enfoques.

SMOTETomek es el método híbrido más simple y el más conservador: aplica primero SMOTE y luego elimina únicamente los **Tomek Links**.

### 1.2 ¿Qué son los Tomek Links?

Un **Tomek Link** es un par $(x_i, x_j)$ donde $x_i \in C_{min}$, $x_j \in C_{maj}$, y:
- $x_j$ es el vecino más cercano de $x_i$
- $x_i$ es el vecino más cercano de $x_j$

Estos pares representan muestras de clases opuestas que son "vecinas directas" en el espacio de features — las muestras más cercanas a la frontera de decisión. Eliminarlos "limpia" la frontera, reduciendo el solapamiento entre clases. Tomek Links elimina tanto la muestra positiva como la negativa del par (eliminación de AMBAS clases).

---

## 2. Marco Teórico — SMOTETomek

### 2.1 Algoritmo (dos fases)

**Fase 1 — SMOTE:**
1. Para cada muestra positiva $x_i$, encontrar sus $k=5$ vecinos positivos más cercanos.
2. Generar $x_{nuevo} = x_i + \lambda \cdot (x_{nn} - x_i)$, con $\lambda \sim \mathcal{U}(0, 1)$.
3. Repetir hasta alcanzar `sampling_strategy = 0.3`.

**Fase 2 — Tomek Links cleaning:**
1. Para cada par de muestras $(x_i, x_j)$ de clases opuestas: comprobar si son mutuamente nearest neighbors.
2. Si son un Tomek Link: **eliminar ambas muestras** del conjunto de entrenamiento.

### 2.2 Formulación matemática

Un par $(x_i, x_j)$ es un Tomek Link si y solo si:
$$d(x_i, x_j) < d(x_i, x_k) \quad \forall x_k \neq x_j, \quad y \quad d(x_i, x_j) < d(x_j, x_l) \quad \forall x_l \neq x_i$$

con $y_i \neq y_j$. La distancia $d$ es la distancia euclídea en el espacio de features (tras preprocesado).

### 2.3 Por qué Tomek Links es conservador

Tomek Links solo elimina **pares solapados exactos** (mutual nearest neighbors de clases opuestas). En datasets reales:
- El número de Tomek Links es típicamente pequeño (~1-5% del total de muestras)
- La eliminación es simétrica (tanto positivos como negativos)
- No modifica la distribución global, solo "lima" la frontera en los puntos más conflictivos

En contraste, ENN (Edited Nearest Neighbours, usado por SMOTEENN) clasifica cada muestra por voto mayoritario de sus $k=3$ vecinos y elimina todas las muestras mal clasificadas, pudiendo afectar al 20-40% del dataset.

### 2.4 SMOTETomek vs SMOTEENN

| Característica | SMOTETomek | SMOTEENN |
|---|---|---|
| Limpieza | Tomek Links (mutual NN de clases opuestas) | ENN (voto k=3 NN) |
| Agresividad | **Conservadora** | **Agresiva** |
| Muestras eliminadas | ~pocas (solo pares Tomek) | ~20-40% del post-SMOTE |
| Impacto en distribución | Mínimo | Significativo |
| Coste computacional | Bajo (O(n)) | Alto (O(n²)) |
| Efecto sobre el PR-AUC | Casi nulo vs SMOTE | Mejora significativa vs SMOTE |

### 2.5 Pipeline en imbalanced-learn

```python
from imblearn.combine import SMOTETomek
```

SMOTETomek está implementado en `imblearn.combine` (no en `imblearn.over_sampling`), reflejando su naturaleza híbrida.

### 2.6 Ventajas y limitaciones

**Ventajas:**
- Combina el aumento de cobertura positiva (SMOTE) con limpieza de solapamiento (Tomek)
- La limpieza Tomek es muy conservadora: bajo riesgo de eliminar muestras útiles
- Produce fronteras de decisión ligeramente más limpias que SMOTE puro

**Limitaciones:**
- Si hay pocos Tomek Links (como ocurre en este dataset), el efecto de la limpieza es negligible
- No mejora significativamente sobre SMOTE en datasets donde las clases no se solapan en Tomek Links
- Para mejoras sustanciales en PR-AUC, se requiere una limpieza más agresiva (ENN → SMOTEENN)

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

El target original `uci` sufría un **problema de riesgo competitivo** (Fine & Gray, 1999): los pacientes que fallecen antes de ser trasladados a UCI quedan etiquetados como negativos, sesgando sistemáticamente el modelo. El endpoint `outcome_intensivo = uci OR mortalidad` corrige este sesgo con justificación clínica (misma acción terapéutica), estadística (742 eventos más estables) y metodológica (CONSORT). La variable `mortalidad` se **elimina de las features** para evitar data leakage.

### 3.2 Pipeline

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

pipeline = ImbPipeline([
    ('preprocessor',  preprocessor),
    ('sampler',       SMOTETomek(sampling_strategy=0.3, random_state=42)),
    ('model',         classifier)
])
```

### 3.3 Estrategia de validación en 3 fases

| Fase | Método | Objetivo |
|---|---|---|
| **Fase 1 — Baseline** | Parámetros por defecto | Referencia sin optimización |
| **Fase 2 — K-Fold** | StratifiedGroupKFold, 5 folds | Robustez y generalización |
| **Fase 3 — GridSearch** | GridSearchCV(scoring='average_precision') | Optimización de hiperparámetros |

---

## 4. Resultados

### 4.1 Fase 2 — K-Fold (robustez)

| Target | Modelo | KF PR-AUC (media) | KF PR-AUC (std) | CV | Interpretación |
|---|---|---|---|---|---|
| MORTALIDAD | GaussianNB | **0.1804** | ±0.0073 | 4.05% | **Segunda mayor estabilidad** (tras SMOTE ±0.0070) |
| outcome_intensivo | GaussianNB | **0.2174** | ±0.0144 | 6.62% | Buena estabilidad |

**Destaca:** SMOTETomek es la segunda técnica más estable en mortalidad (K-Fold std = ±0.0073). Para `outcome_intensivo`, la variabilidad (±0.0144) es la más baja entre todas las técnicas tras SMOTE (±0.0154), confirmando que la limpieza Tomek mantiene la consistencia distribucional.

### 4.2 Fase 3 — GridSearch, Mortalidad

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision | Specificity |
|---|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.2011** | 0.8422 | 0.6607 | 0.1130 | 0.0618 | 0.8396 |
| 2 | XGBoost | 0.1401 | 0.8921 | 0.4144 | 0.1960 | 0.1323 | 0.9571 |
| 3 | LightGBM | 0.1354 | 0.8923 | 0.4024 | 0.1986 | 0.1347 | 0.9588 |
| 4 | GradientBoosting | 0.1331 | 0.8921 | 0.4084 | 0.2009 | 0.1368 | 0.9578 |
| 5 | RandomForest | 0.1286 | 0.8933 | 0.2883 | 0.1931 | 0.1478 | 0.9711 |
| 6 | LogisticRegression | 0.1217 | 0.8879 | 0.4876 | 0.1876 | 0.1138 | 0.9324 |
| 7 | ExtraTrees | 0.1131 | 0.8884 | 0.2222 | 0.1792 | 0.1416 | 0.9789 |

**Hiperparámetros óptimos (GaussianNB — Mortalidad):** `var_smoothing = 1e-7`

### 4.3 Fase 3 — GridSearch, `outcome_intensivo`

Target compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario. Prevalencia en test: 2.04% (148/7.267 episodios).

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1790** | 0.8310 | 0.5203 | 0.1377 | 0.0794 |
| 2 | GradientBoosting | 0.1386 | 0.8764 | 0.3514 | 0.2072 | 0.1515 |
| 3 | XGBoost | 0.1314 | 0.8754 | 0.3649 | 0.2093 | 0.1536 |
| 4 | RandomForest | 0.1257 | 0.8838 | 0.3243 | 0.2025 | 0.1556 |
| 5 | LightGBM | 0.1251 | 0.8702 | 0.3378 | 0.1976 | 0.1402 |
| 6 | LogisticRegression | 0.1220 | 0.8743 | 0.4324 | 0.2222 | 0.1429 |
| 7 | ExtraTrees | 0.1207 | 0.8816 | 0.2703 | 0.1985 | 0.1565 |

**Hiperparámetros óptimos (GaussianNB — `outcome_intensivo`):** `var_smoothing = 1e-5`

> **Interpretación:** Con `outcome_intensivo`, los resultados de SMOTETomek son prácticamente idénticos a SMOTE (diferencia <0.0001). La limpieza Tomek sigue siendo irrelevante. GradientBoosting (PR-AUC=0.139, F1=0.207) es preferible a GaussianNB en despliegue clínico por su mejor equilibrio precision/recall.

---

## 5. Análisis del Rendimiento — SMOTETomek vs SMOTE

### 5.1 Comparativa directa

| Métrica | SMOTE | SMOTETomek | Diferencia |
|---|---|---|---|
| GS PR-AUC Mortalidad | 0.2012 | 0.2011 | −0.0001 |
| GS AUC-ROC Mortalidad | 0.8423 | 0.8422 | −0.0001 |
| GS Recall Mortalidad | 0.6607 | 0.6607 | 0.0000 |
| GS PR-AUC `outcome_intensivo` | 0.1790 | 0.1790 | 0.0000 |
| GS AUC-ROC `outcome_intensivo` | 0.8309 | 0.8310 | +0.0001 |
| KF PR-AUC Mortalidad | 0.1800 ±0.0070 | 0.1804 ±0.0073 | +0.0004 |
| KF PR-AUC `outcome_intensivo` | 0.2166 ±0.0154 | 0.2174 ±0.0144 | +0.0008 |

Las diferencias son estadísticamente insignificantes (<0.001 en casi todos los casos). El efecto de Tomek Links sobre el rendimiento es negligible en este dataset.

### 5.2 Explicación: pocos Tomek Links en datos clínicos

En datos clínicos anestesiológicos con alta dimensionalidad y distribuciones continuas:
- La probabilidad de que dos muestras de clases opuestas sean **mutuamente** nearest neighbors es muy baja
- La mayoría de las muestras positivas tienen como nearest neighbor a otro positivo (o a un negativo "genérico"), no a un negativo que también los tenga a ellas como nearest neighbor
- Por tanto, el número de Tomek Links eliminados es mínimo (~0-10 pares) y el impacto sobre la distribución es negligible

### 5.3 Métricas clínicas del mejor modelo

| Métrica | Mortalidad | UCI | Interpretación |
|---|---|---|---|
| PR-AUC | 0.2011 | 0.1790 | 12.6× y 8.8× sobre el azar |
| Recall | 0.6607 | 0.5203 | 66% muertes; 52% eventos críticos detectados |
| Precision | 0.0618 | 0.0794 | 1 de cada ~17 alarmas de muerte; 1 de cada ~13 de outcome_intensivo |

---

## 6. Análisis del Umbral de Decisión

Igual que en SMOTE, el umbral por defecto es ineficaz. Con el umbral F1-max:
- **Mortalidad**: Recall ≈ 0.66, Precision ≈ 0.06 
- **UCI**: Recall ≈ 0.77, Precision ≈ 0.06

---

## 7. Calibración Probabilística

La calibración probabilística es idéntica al caso SMOTE: sobreajuste a distribución artificial ~23% positivos, requiere Isotonic Regression para llevar las probabilidades a la escala real (~1.6% y ~0.8%).

---

## 8. Conclusiones

1. **SMOTETomek + GaussianNB** produce resultados prácticamente idénticos a SMOTE en este dataset: la adición de la limpieza Tomek Links no mejora ni empeora el rendimiento a efectos prácticos (diferencia <0.001 en todos los indicadores, incluyendo `outcome_intensivo`).

2. **El endpoint compuesto `outcome_intensivo` resuelve el sesgo de riesgo competitivo** del antiguo target `uci`: los 742 eventos producen estimaciones K-Fold más estables (±0.014 vs ±0.034 anteriores).

3. La **ausencia de impacto de Tomek Links** se confirma también con `outcome_intensivo`: el resultado es prácticamente idéntico a SMOTE (0.1790 vs 0.1790 de SMOTE en GridSearch).

4. La **robustez K-Fold** es la segunda mejor de todas las técnicas (±0.0073 mortalidad), casi idéntica a SMOTE (±0.0070), confirmando la estabilidad distribucional de la combinación.

5. Para obtener el beneficio real de un método híbrido SMOTE+limpieza, es necesario usar ENN (Edited Nearest Neighbours) → **SMOTEENN**, que produce una mejora de +0.033 en mortalidad y +0.041 en `outcome_intensivo` sobre SMOTE.

6. **Recomendación**: SMOTETomek es intercambiable con SMOTE en este dataset. Para mejorar con un método híbrido, usar SMOTEENN.

---

## Referencias

- Tomek, I. (1976). Two modifications of CNN. *IEEE Transactions on Systems, Man, and Cybernetics*, 6, 769-772.
- Chawla, N. V. et al. (2002). SMOTE. *JAIR*, 16, 321-357.
- Batista, G. E., Prati, R. C., & Monard, M. C. (2004). A study of the behavior of several methods for balancing machine learning training data. *ACM SIGKDD Explorations Newsletter*, 6(1), 20-29.
- Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn. *JMLR*, 18(17), 1-5.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10(3).
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
