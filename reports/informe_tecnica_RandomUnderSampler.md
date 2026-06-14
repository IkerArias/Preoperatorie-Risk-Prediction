# Informe Técnico — RandomUnderSampler  
## Predicción de Mortalidad y Requerimiento de Cuidados Críticos Postquirúrgicos  
### Trabajo de Fin de Grado — Ingeniería en Ciencia de Datos  

**Técnica de balanceo:** RandomUnderSampler (*Random Under-Sampling*)  
**Tipo:** Undersampling por eliminación aleatoria de la clase mayoritaria  
**Autor:** Iker Arias  
**Fecha:** Mayo 2026  

---

## Resumen Ejecutivo

Se aplica la técnica **RandomUnderSampler** para abordar el severo desequilibrio de clases presente en el dataset clínico anestesiológico (mortalidad postquirúrgica: ~1.6%; requerimiento de cuidados críticos `outcome_intensivo`: ~2.04%). Se evalúan 7 algoritmos de clasificación a través de tres fases de validación (Baseline, K-Fold estratificado y GridSearch), usando PR-AUC como métrica de selección principal.

El mejor modelo para **mortalidad** es **GaussianNB** con PR-AUC = 0.2106 (tercer mejor entre todas las técnicas), AUC-ROC = 0.8342 y Recall = 0.6336. Para **`outcome_intensivo`** (endpoint compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario), también **GaussianNB** con PR-AUC = 0.1485, el **peor resultado para `outcome_intensivo`** entre las 8 técnicas.

**Hallazgo central:** RandomUnderSampler es la única técnica de *undersampling* evaluada. Para mortalidad es sorprendentemente efectiva (tercer mejor PR-AUC). Para `outcome_intensivo`, es la peor técnica evaluada, revelando que la eliminación de negativos destruye información crítica. Es también la técnica significativamente **más rápida**.

> **Nota metodológica:** El segundo target fue redefinido de `uci` (~0.8% prevalencia, 318 eventos) a `outcome_intensivo = (uci==1) | (mortalidad==1)` (~2.04% prevalencia, 742 eventos) para corregir un sesgo de riesgo competitivo. Ver Sección 3.1.

---

## 1. Introducción

### 1.1 Oversampling vs Undersampling

Las estrategias de balanceo se dividen en dos grandes familias:

- **Oversampling**: aumenta las muestras de la clase minoritaria (SMOTE, ADASYN, BorderlineSMOTE, etc.)
- **Undersampling**: reduce las muestras de la clase mayoritaria

RandomUnderSampler es la técnica de undersampling más simple: descarta aleatoriamente muestras negativas hasta alcanzar el ratio deseado. El enfoque es radicalmente diferente al oversampling: en lugar de añadir información, descarta parte del conocimiento disponible.

### 1.2 Motivación y riesgos

La motivación para el undersampling es la velocidad: con $n_{neg}$ >> $n_{pos}$, reducir los negativos puede reducir el tiempo de entrenamiento ×18-20 sin perder (teóricamente) información relevante si los negativos descartados son "redundantes". Sin embargo, el riesgo es evidente: se descarta información real que podría ser necesaria para caracterizar la clase negativa y sus fronteras con la positiva.

---

## 2. Marco Teórico — RandomUnderSampler

### 2.1 Algoritmo

RandomUnderSampler implementa el método más directo de undersampling:

1. Identificar la clase mayoritaria $C_{maj}$ con $n_{maj}$ muestras.
2. Calcular el número de muestras a conservar: $n_{keep} = \lfloor n_{min} / \text{sampling\_strategy} \rfloor$.
3. **Muestrear aleatoriamente sin reemplazamiento** $n_{keep}$ muestras de $C_{maj}$.
4. Descartar el resto de negativos.

El conjunto de entrenamiento resultante tiene $n_{min} + n_{keep}$ muestras.

### 2.2 Formulación matemática

$$n_{keep} = \left\lfloor \frac{n_{min}}{\text{sampling\_strategy}} \right\rfloor$$

Con `sampling_strategy = 1.0` (caso extremo, ratio 1:1): $n_{keep} = n_{min} \approx 128$.  
Con `sampling_strategy = 0.5` (ratio 1:2): $n_{keep} \approx 256$.  
Con `sampling_strategy = 0.3` (ratio 1:3.3): $n_{keep} \approx 427$.

**El GridSearch optimiza `sampling_strategy` ∈ [0.3, 0.5, 1.0]**, lo que permite encontrar el ratio óptimo entre conservar más negativos (mejor caracterización de la clase negativa) y ahorrar tiempo de entrenamiento.

### 2.3 Implicaciones del tamaño del dataset

| sampling_strategy | $n_{neg,keep}$ | $n_{total,train}$ | Factor de reducción |
|---|---|---|---|
| 1.0 (1:1) | 128 | 256 | **~29×** respecto al original |
| 0.5 (1:2) | 256 | 384 | ~20× |
| 0.3 (1:3.3) | 427 | 555 | ~14× |
| Sin RUS | 7,432 | 7,560 | 1× (original) |

Esto explica por qué RandomUnderSampler es la técnica más rápida: con sampling_strategy=0.5, se entrena con solo ~384 muestras en lugar de ~7,560.

### 2.4 Diferencia en el pipeline respecto al oversampling

A diferencia de los notebooks de oversampling, el pipeline de undersampling usa nombres de pasos diferentes:

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

pipeline = ImbPipeline([
    ('preprocessor',  preprocessor),
    ('undersampler',  RandomUnderSampler(random_state=42)),  # → 'undersampler__sampling_strategy' en GS
    ('classifier',    classifier)                             # → 'classifier__' en GS
])
```

En el GridSearch, los parámetros se prefijan con `undersampler__sampling_strategy` y `classifier__`, a diferencia del `sampler__` y `model__` del oversampling.

### 2.5 Ventajas y limitaciones

**Ventajas:**
- Reducción drástica del tiempo de entrenamiento: ×14-29 más rápido que oversampling
- No introduce muestras artificiales: todos los datos en el modelo son reales
- Pequeño footprint de memoria
- El GridSearch de `sampling_strategy` permite encontrar el equilibrio óptimo entre velocidad y rendimiento

**Limitaciones:**
- **Descarta información real**: se elimina potencialmente hasta el 96% de los negativos
- Para targets de prevalencia muy baja (UCI ~0.8%), la pérdida de información negativa es destructiva
- El conjunto de entrenamiento resultante puede ser demasiado pequeño para modelos complejos
- La selección aleatoria no garantiza representatividad de todos los "perfil" de negativos

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

El target original `uci` sufría un **problema de riesgo competitivo** (Fine & Gray, 1999): los pacientes que fallecen antes de ser trasladados a UCI quedan etiquetados como negativos, produciendo un sesgo sistemático que penaliza especialmente a los modelos que aprenden patrones de severidad extrema. El endpoint compuesto `outcome_intensivo = uci OR mortalidad` corrige este sesgo y produce un target con 742 eventos (vs 318), mucho más robusto para el undersampling. La variable `mortalidad` se **elimina de las features** para evitar data leakage.

### 3.2 Particularidades del GridSearch para undersampling

El GridSearch optimiza simultáneamente el ratio de undersampling y los hiperparámetros del clasificador. Para GaussianNB los mejores parámetros encontrados son:

- **Mortalidad**: `undersampler__sampling_strategy = 1.0`, `classifier__var_smoothing = 1e-5`
- **UCI**: `undersampler__sampling_strategy = 0.5`, `classifier__var_smoothing = 1e-9`

El hecho de que la mejor `sampling_strategy` sea distinta para cada target refleja que la frontera de decisión óptima requiere distinto ratio de clase según la prevalencia.

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
| MORTALIDAD | GaussianNB | **0.1955** | ±0.0111 | 5.68% | Alta estabilidad — **Segunda mejor** robustez |
| outcome_intensivo | GaussianNB | **0.1793** | ±0.0431 | 24.04% | **Muy alta variabilidad** — undersampling inestable |

**Hallazgo crítico `outcome_intensivo`**: RandomUnderSampler obtiene PR-AUC K-Fold = 0.1793 para `outcome_intensivo` con variabilidad extrema (±0.043, CV=24%), el peor resultado entre todas las técnicas. Con solo 742 positivos y un undersampling agresivo, el conjunto de entrenamiento por fold es muy pequeño e inestable. La eliminación masiva de negativos impide al modelo caracterizar adecuadamente la clase negativa, especialmente cuando el target es más complejo (endpoint compuesto).

### 4.2 Fase 3 — GridSearch, Mortalidad

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision | Specificity |
|---|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.2106** | 0.8342 | 0.6336 | 0.1114 | 0.0611 | 0.8354 |
| 2 | LightGBM | 0.1414 | 0.8891 | 0.4265 | 0.2043 | 0.1388 | 0.9547 |
| 3 | XGBoost | 0.1362 | 0.8860 | 0.3614 | 0.1917 | 0.1330 | 0.9577 |
| 4 | GradientBoosting | 0.1284 | 0.8829 | 0.3854 | 0.1953 | 0.1335 | 0.9547 |
| 5 | RandomForest | 0.1233 | 0.8808 | 0.2102 | 0.1700 | 0.1383 | 0.9747 |
| 6 | LogisticRegression | 0.1232 | 0.8767 | 0.4819 | 0.1849 | 0.1119 | 0.9313 |
| 7 | ExtraTrees | 0.1079 | 0.8769 | 0.1482 | 0.1401 | 0.1452 | 0.9830 |

**Hiperparámetros óptimos (GaussianNB — Mortalidad):** `undersampler__sampling_strategy = 1.0`, `classifier__var_smoothing = 1e-5`

### 4.3 Fase 3 — GridSearch, `outcome_intensivo`

Target compuesto: ingreso inesperado en UCI **o** fallecimiento intrahospitalario. Prevalencia en test: 2.04% (148/7.267 episodios).

| Rank | Modelo | PR-AUC | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|---|---|
| **1** | **GaussianNB** | **0.1485** | 0.8344 | 0.3716 | 0.1267 | 0.0764 |
| 2 | XGBoost | 0.1337 | 0.8780 | 0.8041 | 0.1569 | 0.0877 |
| 3 | LightGBM | 0.1241 | 0.8784 | 0.8108 | 0.1610 | 0.0903 |
| 4 | LogisticRegression | 0.1216 | 0.8722 | 0.4189 | 0.1845 | 0.1230 |
| 5 | RandomForest | 0.1191 | 0.8873 | 0.4257 | 0.2006 | 0.1484 |
| 6 | ExtraTrees | 0.1175 | 0.8819 | 0.7635 | 0.1638 | 0.0920 |
| 7 | GradientBoosting | 0.1167 | 0.8754 | 0.8176 | 0.1550 | 0.0867 |

**Hiperparámetros óptimos (GaussianNB — `outcome_intensivo`):** `undersampler__sampling_strategy = 0.5`, `classifier__var_smoothing = 1e-9`

> **Interpretación:** El undersampling es especialmente perjudicial para `outcome_intensivo`: todos los modelos tienen PR-AUC por debajo de 0.15, indicando que la pérdida de información negativa es crítica. XGBoost y LightGBM consiguen Recall muy alto (0.80+) a costa de muy baja precisión, produciendo demasiadas falsas alarmas. No se recomienda RandomUnderSampler para este target.

---

## 5. Análisis del Mejor Modelo — GaussianNB + RandomUnderSampler

### 5.1 Métricas clínicas

| Métrica | Mortalidad | UCI | Interpretación clínica |
|---|---|---|---|
| PR-AUC | 0.2106 | 0.1485 | 13.2× y 7.3× sobre el azar |
| AUC-ROC | 0.8342 | 0.8344 | Discriminación global aceptable para mortalidad |
| Recall | 0.6336 | 0.3716 | 63% de muertes detectadas; solo 37% de eventos críticos |
| sampling_strategy óptima | 1.0 (1:1) | 0.5 (1:2) | Más agresivo para mortalidad |

### 5.2 Por qué funciona bien para mortalidad

Con `sampling_strategy = 1.0`, el modelo entrena con solo ~256 muestras (128 pos + 128 neg). Esto parece contraintuitivo, pero para GaussianNB tiene sentido: el modelo solo necesita estimar $\hat{\mu}$ y $\hat{\sigma}^2$ de ambas clases. Con muestras perfectamente balanceadas, las estimaciones son más simétricas y la probabilidad posterior $P(y=1|x)$ está mejor calibrada en un dataset balanceado.

### 5.3 Por qué falla para UCI

Para UCI, la prevalencia del ~0.8% significa que hay ~64 positivos en el train. Con `sampling_strategy=0.5`, se conservan ~128 negativos → ~192 muestras totales. Con tan pocos datos, las estimaciones de GaussianNB son inestables y la representación de los negativos es insuficiente para caracterizar la clase. El K-Fold PR-AUC de 0.1531 confirma esta degradación sistemática.

### 5.4 Comparativa velocidad vs rendimiento

| Aspecto | RandomUnderSampler | SMOTE | SMOTEENN |
|---|---|---|---|
| Tiempo de entrenamiento | ~5-10s | ~60-120s | ~300-600s |
| PR-AUC mortalidad | 0.2106 | 0.2012 | **0.2346** |
| PR-AUC `outcome_intensivo` | 0.1485 | 0.1790 | **0.2203** |
| Factor velocidad | **×1 (más rápido)** | ~×12 más lento | ~×60 más lento |

RandomUnderSampler ofrece el **mejor ratio rendimiento/tiempo** para mortalidad: mejora a SMOTE en PR-AUC siendo ~12× más rápido. Sin embargo, para UCI, la degradación de PR-AUC no es aceptable en un contexto clínico.

---

## 6. Análisis del Umbral de Decisión

| Umbral | Mortalidad Recall | Mortalidad Precision | UCI Recall | UCI Precision |
|---|---|---|---|---|
| 0.5 (defecto) | ~0.00 | N/A | ~0.00 | N/A |
| F1-max | ~0.63 | ~0.06 | ~0.63 | ~0.04 |
| Recall ≥ 90% | ≥0.90 | ~0.03 | ≥0.90 | ~0.02 |

La menor precision para UCI (0.04 con F1-max) respecto a otras técnicas (~0.05-0.06) refleja el rendimiento más débil del modelo en ese target.

---

## 7. Calibración Probabilística

RandomUnderSampler con `sampling_strategy=1.0` entrena con 50% de positivos. Las probabilidades de salida estarán masivamente infladas respecto a la prevalencia real (~1.6%). La calibración es **más crítica** aquí que para otras técnicas: las probabilidades sin calibrar pueden ser hasta 30× mayores que las reales. Se aplica Isotonic Regression sobre el test set.

---

## 8. Conclusiones

1. **RandomUnderSampler + GaussianNB** logra el **tercer mejor PR-AUC en mortalidad** (0.2106) a pesar de ser la técnica más simple de undersampling, con el mayor ahorro computacional de todas las técnicas evaluadas.

2. **El endpoint compuesto `outcome_intensivo` resuelve el sesgo de riesgo competitivo** del antiguo target `uci`, pero el undersampling sigue siendo perjudicial para este target: PR-AUC=0.1485, el peor entre todas las técnicas, con variabilidad K-Fold extrema (±0.043).

3. La **velocidad** es la ventaja distintiva: ~18-20× más rápido que SMOTE y ~60× más rápido que SMOTEENN. Para sistemas con restricciones de latencia en el target de mortalidad, puede ser la elección más práctica.

4. Los parámetros óptimos (`sampling_strategy = 1.0` para mortalidad) revelan que un dataset perfectamente balanceado 1:1 es suficiente para GaussianNB en mortalidad, pero insuficiente para `outcome_intensivo`.

5. **Recomendación**: RandomUnderSampler es recomendable solo para el target de mortalidad con restricciones computacionales. Para `outcome_intensivo`, cualquier técnica de oversampling es significativamente superior.

---

## Referencias

- Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn. *JMLR*, 18(17), 1-5.
- Liu, X. Y., Wu, J., & Zhou, Z. H. (2009). Exploratory Undersampling for Class-Imbalance Learning. *IEEE Transactions on Systems, Man, and Cybernetics*, 39(2), 539-550.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLOS ONE*, 10(3).
- He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS 2017*.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
- Freemantle, N., Calvert, M., Wood, J., et al. (2003). Composite outcomes in randomized trials. *JAMA*, 289(19), 2554-2559.
